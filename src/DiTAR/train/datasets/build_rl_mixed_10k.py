#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the mixed RL fine-tuning dataset used by GROW (ORW / SDE).

Note on the name: "10k" is the per-group sample count. The merged output has
three groups (Emilia-EN, Emilia-ZH, LibriTTS-EN) of 10k rows each, i.e. 30k rows
total (see ``--n_per_lang`` / ``--libritts_n``).

Produces a **single** merged dataset whose schema matches the pretrain data
plus one extra ``language`` column:

    columns = [audio_path, text, gen_text, duration, language]
    duration: float64 ; language: str ("en" / "zh")

The existing loader (``load_ldm_dataset``, ``audio_type=vae_online``) keeps only
the columns it needs, so the extra ``language`` column is harmless and the ORW /
SDE configs need no change -- just point ``datasets.train_ds_path`` at the
output directory of this script (default ``data/RL_mixed_emilia_libritts_30k``).

Dataset composition (all merged into ONE dataset, 30k rows total):
    - Emilia EN  : 10k rows
    - Emilia ZH  : 10k rows
    - LibriTTS EN: 10k rows (from train-clean-100)

Each row:
    - audio_path : prompt speech (voice to clone)
    - text       : the prompt's own transcript
    - gen_text   : target text to synthesize, drawn from a real recording in the
                   SAME group (same corpus + language)
    - duration   : prompt audio duration (float)
    - language   : "en" / "zh" (Emilia inferred from path; LibriTTS defaults to en)

Difficulty design:
    gen_text is always a real recording's transcript (each in [1.5, max_gen_dur]s,
    never concatenated / never artificially long). To create reward variance we
    oversample longer (harder) targets by bucketing on the SOURCE recording's
    duration into two bands:
      - with prob ``long_frac`` pick from the long band ((p80, max_gen_dur]);
        otherwise pick from the short band (<= p80).

Durations are kept in [1.5, 15]s; ``long_frac`` defaults to 0.3.

Usage:
    # full build (single dataset, 30k rows total):
    python build_rl_mixed_10k.py

    # small dry-run self-check:
    python build_rl_mixed_10k.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Dict, List

import numpy as np

# CJK ranges (for Chinese length statistics only)
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _lang_of(path: str) -> str:
    if "/EN/" in path:
        return "EN"
    if "/ZH/" in path:
        return "ZH"
    return "?"


def _textlen(text: str, lang: str) -> int:
    """Text length (for summary display only): EN = word count, ZH = CJK char count."""
    if lang == "ZH":
        return len(_CJK.findall(text))
    return len(text.split())


# =========================================================================
# Difficulty design: assign a gen_text to each prompt
# =========================================================================
def assign_gen_text(prompts: List[dict], pool: List[dict], lang: str, src_tag: str, args, rng: random.Random):
    """Pair each prompt with a gen_text drawn from the same-group ``pool``.

    gen_text is always a real recording's transcript (each in [1.5, max_gen_dur]s,
    never concatenated / never artificially long), so target synthesis duration is
    naturally <= max_gen_dur. Difficulty = the source recording's duration:
      - with prob ``long_frac`` draw from the long band ((p80, max_gen_dur]);
        otherwise draw from the short band (<= p80).
    """
    src = [r for r in pool if r["duration"] <= args.max_gen_dur]
    if not src:
        src = pool
    durs = np.array([r["duration"] for r in src])
    p80 = float(np.percentile(durs, 80))
    short_pool = [r for r in src if r["duration"] <= p80]
    long_pool = [r for r in src if r["duration"] > p80]
    if not short_pool:
        short_pool = src
    if not long_pool:
        long_pool = sorted(src, key=lambda r: -r["duration"])[: max(1, len(src) // 5)]

    out = []
    n_long = 0
    for r in prompts:
        is_long = rng.random() < args.long_frac
        src_pool = long_pool if is_long else short_pool
        g = rng.choice(src_pool)
        gen = g["text"]
        # avoid gen_text being identical to the prompt's own text
        tries = 0
        while gen.strip() == r["text"].strip() and tries < 5:
            g = rng.choice(src_pool)
            gen = g["text"]
            tries += 1
        out.append(
            {
                "audio_path": r["audio_path"],
                "text": r["text"],
                "gen_text": gen,
                # duration is float64, matching the pretrain dataset schema
                "duration": float(r["duration"]),
                # language field (Emilia inferred from path EN/ZH, LibriTTS defaults EN), lowercased
                "language": lang.lower(),
                "_lang": lang,
                "_src": src_tag,
                "_is_long": is_long,
                "_gen_dur": g["duration"],   # proxy for target synthesis duration = gen_text source recording duration
                "_genlen": _textlen(gen, lang),
            }
        )
        n_long += int(is_long)
    return out, n_long


# =========================================================================
# 1) Emilia pool
# =========================================================================
def collect_pool_emilia(d, args) -> Dict[str, List[dict]]:
    """Random sampling + vectorized filtering to collect a per-language pool."""
    import pandas as pd

    N = len(d)
    rng = np.random.default_rng(args.seed)
    n_draw = min(args.draw_indices, N)
    idx = rng.choice(N, size=n_draw, replace=False)

    pools: Dict[str, List[dict]] = {"EN": [], "ZH": []}
    target = args.pool_per_lang
    chunk = 50000
    t0 = time.time()
    seen = 0
    for s in range(0, n_draw, chunk):
        chunk_idx = idx[s : s + chunk].tolist()
        sub = d.select(chunk_idx)
        df = sub.to_pandas()
        seen += len(df)

        df["dur_f"] = pd.to_numeric(df["duration"], errors="coerce")
        df["dns_f"] = pd.to_numeric(df["dnsmos"], errors="coerce")
        df = df.dropna(subset=["dur_f", "dns_f"])
        df = df[
            (df["dur_f"] >= args.min_dur)
            & (df["dur_f"] <= args.max_dur)
            & (df["dns_f"] >= args.min_dnsmos)
        ]
        if len(df) == 0:
            continue
        df["lang"] = df["audio_path"].map(_lang_of)
        df = df[df["lang"].isin(("EN", "ZH"))]
        df = df[df["text"].astype(str).str.strip().str.len() > 0]

        for lang in ("EN", "ZH"):
            if len(pools[lang]) >= target:
                continue
            sl = df[df["lang"] == lang]
            for ap, tx, du in zip(sl["audio_path"], sl["text"], sl["dur_f"]):
                if len(pools[lang]) >= target:
                    break
                tx = str(tx).strip()
                pools[lang].append(
                    {
                        "audio_path": str(ap),
                        "text": tx,
                        "duration": float(du),
                        "textlen": _textlen(tx, lang),
                    }
                )

        done = all(len(pools[l]) >= target for l in ("EN", "ZH"))
        print(
            f"  [emilia collect] scanned={seen:,} | EN={len(pools['EN']):,} ZH={len(pools['ZH']):,} "
            f"| {time.time()-t0:.0f}s",
            flush=True,
        )
        if done:
            break

    for lang in ("EN", "ZH"):
        if len(pools[lang]) < args.n_per_lang:
            print(
                f"  [WARN] Emilia {lang} pool has only {len(pools[lang]):,} < needed {args.n_per_lang:,}; "
                f"increase --draw_indices or --pool_per_lang.",
                flush=True,
            )
    return pools


# =========================================================================
# 2) LibriTTS train-clean-100 pool (English)
# =========================================================================
def _read_one_libritts(wav_path: str):
    """Read one LibriTTS item: (audio_path, text, duration) or None. text = normalized."""
    import soundfile as sf

    txt_path = wav_path[:-4] + ".normalized.txt"
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        info = sf.info(wav_path)
        dur = info.frames / info.samplerate
    except Exception:
        return None
    return {"audio_path": wav_path, "text": text, "duration": float(dur)}


def collect_pool_libritts(args) -> List[dict]:
    """Scan all wavs under train-clean-100, filter to [min_dur, max_dur], collect EN pool."""
    base = args.libritts_src
    print(f"  scanning wav list: {base}", flush=True)
    wavs = glob(os.path.join(base, "*", "*", "*.wav"))
    rng = random.Random(args.seed)
    rng.shuffle(wavs)
    if args.dry_run:
        wavs = wavs[:3000]
    print(f"  {len(wavs):,} wavs total, reading headers (multi-threaded) ...", flush=True)

    pool: List[dict] = []
    t0 = time.time()
    target = args.libritts_pool
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_read_one_libritts, wavs), 1):
            if rec is not None and args.min_dur <= rec["duration"] <= args.max_dur:
                rec["textlen"] = _textlen(rec["text"], "EN")
                pool.append(rec)
            if i % 5000 == 0:
                print(
                    f"    scanned={i:,}/{len(wavs):,} | pool={len(pool):,} | {time.time()-t0:.0f}s",
                    flush=True,
                )
            if len(pool) >= target:
                print(f"    reached pool cap {target:,}, stop scanning.", flush=True)
                break

    print(f"  LibriTTS pool: {len(pool):,} rows (in [{args.min_dur},{args.max_dur}]s)", flush=True)
    if len(pool) < args.libritts_n:
        print(f"  [WARN] LibriTTS pool {len(pool):,} < needed {args.libritts_n:,}; relax duration or raise --libritts_pool.", flush=True)
    return pool


# =========================================================================
# summary / checks
# =========================================================================
def audio_check(rows: List[dict], n: int = 6):
    print("\n=== AUDIO CHECK (sample a few audio_path, verify loadable) ===", flush=True)
    ok = 0
    for r in rows[:n]:
        p = r["audio_path"]
        exists = os.path.exists(p)
        loadable = False
        try:
            import torchaudio

            if exists:
                info = torchaudio.info(p)
                loadable = info.num_frames > 0
        except Exception as e:
            print(f"  load err: {e}", flush=True)
        ok += int(exists and loadable)
        print(f"  exists={exists} loadable={loadable} | {p}", flush=True)
    if ok == 0:
        print("  [WARN] none loadable! check that paths / mounts are correct.", flush=True)


def summarize(rows: List[dict]):
    print("\n========================= SUMMARY =========================", flush=True)
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['_src']}-{r['_lang']}", []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        lang = rs[0]["_lang"]
        unit = "words" if lang == "EN" else "chars"
        n_long = sum(int(r["_is_long"]) for r in rs)
        gen_durs = np.array([r["_gen_dur"] for r in rs])
        genlens = np.array([r["_genlen"] for r in rs])
        hours = sum(float(r["duration"]) for r in rs) / 3600.0
        print(
            f"{key}: n={len(rs):,} (long {n_long:,}={n_long/max(len(rs),1):.0%}) | prompt {hours:.1f}h",
            flush=True,
        )
        print(
            f"     gen_dur: min={gen_durs.min():.1f}s p50={np.percentile(gen_durs,50):.1f}s "
            f"p95={np.percentile(gen_durs,95):.1f}s max={gen_durs.max():.1f}s "
            f"| gen_text len p50={np.percentile(genlens,50):.0f}{unit} "
            f"p95={np.percentile(genlens,95):.0f}{unit}",
            flush=True,
        )
    tot_hours = sum(float(r["duration"]) for r in rows) / 3600.0
    print(f"TOTAL n={len(rows):,} | prompt~={tot_hours:.1f}h", flush=True)
    print("\n--- 3 sample rows ---", flush=True)
    for r in rows[:3]:
        print(f"  [{r['_src']}-{r['_lang']}] lang={r['language']} dur={r['duration']:.2f}s", flush=True)
        print(f"    text    : {r['text'][:70]}", flush=True)
        print(f"    gen_text: {r['gen_text'][:90]}", flush=True)
    print("===========================================================", flush=True)


# =========================================================================
def build(args):
    from datasets import Dataset, load_from_disk

    rng = random.Random(args.seed)
    all_rows: List[dict] = []

    # ---- 1) Emilia EN + ZH ----
    print(f"\n############ EMILIA (EN {args.n_per_lang:,} + ZH {args.n_per_lang:,}) ############", flush=True)
    print(f"[emilia] load source: {args.emilia_src}", flush=True)
    d = load_from_disk(args.emilia_src)
    print(f"         total utts = {len(d):,} | cols = {d.column_names}", flush=True)
    print(
        f"[emilia] collect pool (dur in [{args.min_dur},{args.max_dur}]s, dnsmos>={args.min_dnsmos}) "
        f"target {args.pool_per_lang:,}/lang ...",
        flush=True,
    )
    pools = collect_pool_emilia(d, args)
    for lang in ("EN", "ZH"):
        pool = pools[lang]
        rng.shuffle(pool)
        chosen = pool[: min(args.n_per_lang, len(pool))]
        rows, n_long = assign_gen_text(chosen, pool, lang, "emilia", args, rng)
        all_rows += rows
        print(f"  [emilia-{lang}] +{len(rows):,} rows (long {n_long:,})", flush=True)

    # ---- 2) LibriTTS EN ----
    print(f"\n############ LibriTTS train-clean-100 (EN {args.libritts_n:,}) ############", flush=True)
    lt_pool = collect_pool_libritts(args)
    rng.shuffle(lt_pool)
    chosen = lt_pool[: min(args.libritts_n, len(lt_pool))]
    rows, n_long = assign_gen_text(chosen, lt_pool, "EN", "libritts", args, rng)
    all_rows += rows
    print(f"  [libritts-EN] +{len(rows):,} rows (long {n_long:,})", flush=True)

    # ---- merge + save (single dataset) ----
    rng.shuffle(all_rows)
    cols = ["audio_path", "text", "gen_text", "duration", "language"]
    ds = Dataset.from_dict({c: [r[c] for r in all_rows] for c in cols})

    print(f"\n[save] -> {args.out}  (single dataset, {len(all_rows):,} rows total)", flush=True)
    ds.save_to_disk(args.out)

    # duration.json: float, in the SAME row order as save_to_disk (otherwise the
    # dynamic-batch sampler mismatches and OOMs).
    durations = [float(r["duration"]) for r in all_rows]
    with open(os.path.join(args.out, "duration.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f)
    print(
        f"       duration.json written (n={len(durations)}, "
        f"min={min(durations):.2f}s max={max(durations):.2f}s)",
        flush=True,
    )

    audio_check(all_rows)
    summarize(all_rows)
    print("\nNext: point config datasets.train_ds_path at:", flush=True)
    print(f"    {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/RL_mixed_emilia_libritts_30k", help="output dir of the single merged dataset")

    # ---- shared duration / difficulty (all three groups) ----
    ap.add_argument("--min_dur", type=float, default=1.5, help="min prompt audio duration")
    ap.add_argument("--max_dur", type=float, default=15.0, help="max prompt audio duration")
    ap.add_argument("--max_gen_dur", type=float, default=15.0, help="max gen_text source recording duration")
    ap.add_argument("--long_frac", type=float, default=0.3, help="fraction of longer (harder) targets")
    ap.add_argument("--seed", type=int, default=42)

    # ---- Emilia ----
    ap.add_argument("--emilia_src", default="data/Emilia_ZH_EN_char", help="Emilia source with a dnsmos column")
    ap.add_argument("--n_per_lang", type=int, default=10000, help="Emilia rows per language (EN / ZH each)")
    ap.add_argument("--pool_per_lang", type=int, default=30000, help="Emilia candidate pool per language (also the gen_text pool)")
    ap.add_argument("--draw_indices", type=int, default=300000, help="how many Emilia rows to randomly sample for filtering")
    ap.add_argument("--min_dnsmos", type=float, default=3.2, help="Emilia dnsmos threshold")

    # ---- LibriTTS ----
    ap.add_argument("--libritts_src", default="data/LibriTTS/train-clean-100", help="LibriTTS train-clean-100 root")
    ap.add_argument("--libritts_n", type=int, default=10000, help="LibriTTS English rows")
    ap.add_argument("--libritts_pool", type=int, default=30000, help="LibriTTS candidate pool cap (also the gen_text pool)")
    ap.add_argument("--workers", type=int, default=16, help="threads for reading wav headers")

    ap.add_argument("--dry_run", action="store_true", help="small-scale self-check")
    args = ap.parse_args()

    if args.dry_run:
        args.n_per_lang = 100
        args.pool_per_lang = 800
        args.draw_indices = 30000
        args.libritts_n = 100
        args.libritts_pool = 800
        args.out = "data/RL_mixed_emilia_libritts_30k_dryrun"
        print("[DRY-RUN] small-scale self-check mode", flush=True)

    build(args)


if __name__ == "__main__":
    main()

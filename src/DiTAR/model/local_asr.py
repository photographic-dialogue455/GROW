from __future__ import annotations

import string
from typing import List

import torch

import logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
# ---------------------------------------------------------------------------
# WER text normalisation (mirrors seed-tts-eval / utils_eval.run_asr_wer)
# ---------------------------------------------------------------------------


def _get_punctuation_all() -> str:
    # zhon may not be installed everywhere; fall back to ASCII punctuation only.
    try:
        from zhon.hanzi import punctuation as zh_punct  # type: ignore
    except Exception:
        zh_punct = ""
    return zh_punct + string.punctuation


_PUNCT_ALL = _get_punctuation_all()


def _normalise_text(s, lang: str) -> str:
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s) #'But then my longing grew again irresistible, until on its wings I sank back into your arms.'
    elif not isinstance(s, str):
        s = str(s)
    for ch in _PUNCT_ALL:
        s = s.replace(ch, "")
    s = s.replace("  ", " ").strip()
    if lang == "en":
        s = s.lower()
    return s


def _wer_jiwer(truth: str, hypo: str, lang: str) -> float:
    """Standard jiwer-based WER, matching ``utils_eval.run_asr_wer``."""
    from jiwer import compute_measures

    truth = _normalise_text(truth, lang)
    hypo = _normalise_text(hypo, lang)

    if lang == "zh":
        # Char-level for Chinese: split on every character.
        truth = " ".join(list(truth))
        hypo = " ".join(list(hypo))

    measures = compute_measures(truth, hypo)
    wer = float(measures["wer"])
    return max(0.0, min(1.0, wer))


# ---------------------------------------------------------------------------
# Eager ASR loader
# ---------------------------------------------------------------------------


def build_asr_model(lang: str, ckpt_dir: str, device):
    device_str = str(device)
    if lang == "zh":
        from funasr import AutoModel
        return AutoModel(
            model="checkpoints/funasr/paraformer-zh",
            disable_update=True,
            device=device_str,
        )
    elif lang == "en":
        from faster_whisper import WhisperModel
        # faster-whisper picks its own GPU index.
        gpu_idx = 0
        if "cuda" in device_str:
            parts = device_str.split(":")
            if len(parts) == 2:
                try:
                    gpu_idx = int(parts[1])
                except ValueError:
                    gpu_idx = 0
        return WhisperModel(
            "checkpoints/faster-whisper-large-v3",
            device="cuda" if "cuda" in device_str else "cpu",
            device_index=[gpu_idx] if "cuda" in device_str else None,
            compute_type="float16" if "cuda" in device_str else "int8",
        )
    else:
        raise ValueError(
            f"Unsupported ASR language '{lang}'. Use 'zh' or 'en'."
        )


# ---------------------------------------------------------------------------
# Public batch entry point
# ---------------------------------------------------------------------------


def _zero_reward() -> dict:
    return {"score": 0.0, "wer": 1.0, "asr": ""}


@torch.no_grad()
def run_asr_local_batch(
    audios: List[torch.Tensor],
    ref_texts: List[str],
    # source_sr: int = 24000,
    lang: str = "zh",
    model=None,
) -> List[dict]:
    """Transcribe a list of audio tensors and return WER-based rewards.

    Each ``audios[i]`` is a 1-D or (1, T) waveform tensor at ``source_sr``
    (typically 24 kHz, the BigVGAN-VAE output). Tensors may be on GPU; we
    resample to 16 kHz on whichever device they live on, then pass numpy
    arrays to the ASR backend.
    """
    if len(audios) == 0:
        return []
    assert len(audios) == len(ref_texts), (
        f"audios ({len(audios)}) and ref_texts ({len(ref_texts)}) length mismatch"
    )

    waves_np = []
    for a in audios:
        if a.dim() == 2:
            a = a.squeeze(0)
        elif a.dim()==3:
            a = a.squeeze(0).squeeze(0)
        a = a.float()
        # if resampler is not None:
        #     a = resampler(a)
        waves_np.append(a.detach().contiguous().cpu().numpy())

    # Run the ASR backend.
    out: List[dict] = []
    if lang == "zh":
        try:
            try:
                results = model.generate(input=waves_np, batch_size_s=300, disable_pbar=True)
            except TypeError:
                # Older funasr versions don't support those kwargs.
                results = model.generate(input=waves_np)
        except Exception as e:
            print(f"[local_asr] funasr generate failed: {e}; returning zero rewards.")
            return [_zero_reward() for _ in audios]

        # Optional zhconv to convert traditional → simplified, matching seed-tts-eval.
        try:
            import zhconv  # type: ignore

            _zhconv = lambda s: zhconv.convert(s, "zh-cn")  # noqa: E731
        except Exception:
            _zhconv = lambda s: s  # noqa: E731

        for i, ref in enumerate(ref_texts):
            if i >= len(results):
                out.append(_zero_reward())
                continue
            r = results[i]
            hyp = r.get("text", "") if isinstance(r, dict) else str(r)
            hyp = _zhconv(hyp)
            try:
                wer = _wer_jiwer(ref, hyp, "zh")
            except Exception:
                wer = 1.0
            out.append({"score": 1.0 - wer, "wer": wer, "asr": hyp})

    elif lang == "en":
        # faster-whisper transcribes one waveform at a time.
        for i, ref in enumerate(ref_texts):
            try:
                segments, _ = model.transcribe(
                    waves_np[i], beam_size=5, language="en"
                )
                hyp = " ".join(seg.text for seg in segments).strip()
            except Exception as e:
                print(f"[local_asr] faster-whisper failed on item {i}: {e}")
                out.append(_zero_reward())
                continue
            try:
                wer = _wer_jiwer(ref, hyp, "en")
            except Exception:
                wer = 1.0
            out.append({"score": 1.0 - wer, "wer": wer, "asr": hyp})

    else:
        raise ValueError(f"Unsupported ASR language '{lang}'.")

    return out

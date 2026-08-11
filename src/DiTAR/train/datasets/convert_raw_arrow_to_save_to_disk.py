#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert an F5-TTS ``raw.arrow`` into the HuggingFace ``save_to_disk`` format.

The F5-TTS pretraining pipeline (``prepare_emilia_v2.py``) writes a single
``raw.arrow`` file (via ``datasets.arrow_writer.ArrowWriter``) plus a
``duration.json`` / ``vocab.txt``. GROW's training loader
(``DiTAR.model.ldm_dataset.load_ldm_dataset``) instead calls
``datasets.load_from_disk``, which expects the ``save_to_disk`` directory layout
(``data-*.arrow`` + ``dataset_info.json`` + ``state.json``).

This one-liner bridges the two: it reads ``<src>/raw.arrow`` and re-saves it to
``<dst>`` in the ``save_to_disk`` format. Copy the ``duration.json`` produced by
``prepare_emilia_v2.py`` into ``<dst>`` afterwards (the loader reads it there;
it falls back to the dataset's own ``duration`` column if absent).

The dataset schema is unchanged: ``{audio_path, text, duration}``.

Usage:
    python convert_raw_arrow_to_save_to_disk.py \
        --src data/Emilia_ZH_EN_char_raw \
        --dst data/Emilia_ZH_EN_char
"""
import argparse
import os
import shutil

from datasets import Dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir containing raw.arrow (prepare_emilia_v2.py output)")
    ap.add_argument("--dst", required=True, help="output dir in save_to_disk format (load_from_disk expects this)")
    args = ap.parse_args()

    ds = Dataset.from_file(os.path.join(args.src, "raw.arrow"))
    ds.save_to_disk(args.dst)
    print(f"[save_to_disk] {len(ds):,} rows | cols={ds.column_names} -> {args.dst}")

    # carry over duration.json if present (the loader reads it from <dst>)
    src_dur = os.path.join(args.src, "duration.json")
    if os.path.exists(src_dur):
        shutil.copy(src_dur, os.path.join(args.dst, "duration.json"))
        print(f"[duration.json] copied {src_dur} -> {args.dst}/duration.json")


if __name__ == "__main__":
    main()

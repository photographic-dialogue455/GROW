#!/bin/bash
# =============================================================================
#  GROW — single-machine 8-GPU evaluation (reproduces paper Table 1)
# =============================================================================
#  Evaluates the three released checkpoints on the three zero-shot TTS test
#  sets and reports WER / SIM / UTMOS, exactly matching the paper protocol:
#
#     NFE=32 · Euler · sway-sampling coef -1 · CFG=2.0 (sample_strategy=stage)
#     seed=666 · Semantic-VAE vocoder · batch size 1
#
#  The staged CFG schedule (sample_strategy=stage) is the sampler used for both
#  on-policy rollout and evaluation in the paper; it is what produced the Table 1
#  numbers below. A constant-CFG variant (cfg=1.5, sample_strategy=base) is an
#  ablation only and yields different numbers — do NOT use it to reproduce Table 1.
#
#  Released checkpoints (GROW/model_ckpts/), corresponding to Table 1:
#     01_pretrain          model_200000.pt   Table 1 row 1  (Pretrain)
#     02_ditar_grpo_nfe10  model_750.pt      Table 1 row 3  (DiTAR-GRPO, NFE=10)
#     03_grow_nfe10        model_750.pt      Table 1 row 6  (GROW,       NFE=10)
#
#  Test sets (full):  LibriSpeech-PC test-clean (1127) · Seed-TTS EN (1088) ·
#                     Seed-TTS ZH (2020).
#
#  Each (checkpoint × test set) job runs 8-GPU distributed inference, then
#  SIM → UTMOS → WER. Jobs run SEQUENTIALLY, each grabbing all 8 GPUs.
#  Idempotent / resumable: complete wavs and existing metric .jsonl are skipped,
#  so you can safely re-run after an interruption.
#
#  Usage:
#     bash eval_8gpu.sh                 # all 3 checkpoints, all 3 test sets
#     CKPTS="03_grow_nfe10" bash eval_8gpu.sh          # one checkpoint
#     TESTSETS="ls" bash eval_8gpu.sh                  # one test set
#     CKPTS="02_ditar_grpo_nfe10 03_grow_nfe10" TESTSETS="ls en zh" bash eval_8gpu.sh
#
#  Required data (see src/DiTAR/eval/README.md):
#     LIBRISPEECH_DIR  -> LibriSpeech test-clean root (contains <spk>/<chapter>/*.flac)
#     SEEDTTS_DIR      -> seed-tts-eval testset root  (contains en/ zh/ with meta.lst)
#     Manifests: data/librispeech_pc_test_clean_cross_sentence.lst
#                $SEEDTTS_DIR/{en,zh}/meta.lst
# =============================================================================
set -u

# ----------------------------------------------------------------------------
# Paths — override via environment variables as needed.
# ----------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR" || { echo "cannot cd $REPO_DIR"; exit 1; }

CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/model_ckpts}"
# LibriSpeech test-clean root (holds <spk>/<chapter>/<utt>.flac):
LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-$REPO_DIR/data/LibriSpeech/test-clean}"
# seed-tts-eval testset root (holds en/ and zh/, each with meta.lst + prompt-wavs/):
SEEDTTS_DIR="${SEEDTTS_DIR:-$REPO_DIR/data/seedtts_testset}"
# Full LibriSpeech-PC cross-sentence manifest (1127 pairs):
LS_METALST="${LS_METALST:-$REPO_DIR/data/librispeech_pc_test_clean_cross_sentence.lst}"

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/eval_logs}"
mkdir -p "$LOG_DIR"

# Which checkpoints / test sets to run (space separated).
CKPTS="${CKPTS:-01_pretrain 02_ditar_grpo_nfe10 03_grow_nfe10}"
TESTSETS="${TESTSETS:-ls en zh}"

# ----------------------------------------------------------------------------
# GPU / distributed config (single machine, 8 GPUs).
# ----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_NUMS="${GPU_NUMS:-8}"
MAIN_PORT="${MAIN_PORT:-29500}"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/src/DiTAR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ----------------------------------------------------------------------------
# Fixed inference config — matches the paper exactly. Do not change to
# reproduce Table 1.
# ----------------------------------------------------------------------------
NFE_STEP=32
ODE_METHOD="euler"
SWAY_SAMPLING="-1"
CFG_STRENGTH="2.0"
SAMPLE_STRATEGY="stage"    # staged CFG schedule (paper default; reproduces Table 1)
MEL_SPEC_TYPE="semanticvae"
INFER_BATCH_SIZE="1"
SEED=666
TASK="cross_sentence"   # only used in the GEN_PATH naming below (must match the Python output_dir suffix)
TAG="table1"

INFER_PY="src/DiTAR/eval/eval_infer_batch_ditar_parallel.py"
LS_METRIC_PY="src/DiTAR/eval/eval_librispeech_test_clean.py"
SEED_METRIC_PY="src/DiTAR/eval/eval_seedtts_testset.py"
UTMOS_PY="src/DiTAR/eval/eval_utmos.py"

# ----------------------------------------------------------------------------
# checkpoint dir name -> ckpt file name (release layout under $CKPT_ROOT)
# ----------------------------------------------------------------------------
ckpt_file_for() {
  case "$1" in
    01_pretrain)          echo "model_200000.pt" ;;
    02_ditar_grpo_nfe10)  echo "model_750.pt" ;;
    03_grow_nfe10)        echo "model_750.pt" ;;
    *)                    echo "" ;;   # unknown -> auto-detect below
  esac
}

# ----------------------------------------------------------------------------
# Run one (checkpoint, testset) job: 8-GPU inference -> metrics. Idempotent.
#   $1 = ckpt dir name   $2 = testset key (ls|en|zh)
# ----------------------------------------------------------------------------
run_job() {
  local CK="$1" TS="$2"
  local CKDIR="$CKPT_ROOT/$CK"
  local CFG_PATH="$CKDIR/config.yaml"

  local CKFILE; CKFILE="$(ckpt_file_for "$CK")"
  if [ -z "$CKFILE" ]; then
    CKFILE="$(cd "$CKDIR" 2>/dev/null && ls -1 model_*.pt 2>/dev/null | head -1)"
  fi
  local CKPT_PATH="$CKDIR/$CKFILE"

  # per-testset knobs
  local TESTSET METALST PROMPT_DIR METRIC_PY LANG EXPECT
  case "$TS" in
    ls) TESTSET="ls_pc_test_clean"; METALST="$LS_METALST";               PROMPT_DIR="$LIBRISPEECH_DIR"; METRIC_PY="$LS_METRIC_PY";   LANG="en" ;;
    en) TESTSET="seedtts_test_en";  METALST="$SEEDTTS_DIR/en/meta.lst";   PROMPT_DIR="$SEEDTTS_DIR";     METRIC_PY="$SEED_METRIC_PY"; LANG="en" ;;
    zh) TESTSET="seedtts_test_zh";  METALST="$SEEDTTS_DIR/zh/meta.lst";   PROMPT_DIR="$SEEDTTS_DIR";     METRIC_PY="$SEED_METRIC_PY"; LANG="zh" ;;
    *)  echo "unknown testset '$TS' (use ls|en|zh)"; return 1 ;;
  esac

  local GEN_PATH="$RESULTS_DIR/$CK/$TESTSET/seed${SEED}_${ODE_METHOD}_nfe${NFE_STEP}_${MEL_SPEC_TYPE}_ss${SWAY_SAMPLING}_cfg${CFG_STRENGTH}_bsz${INFER_BATCH_SIZE}_${TAG}_${TASK}"
  local LOG="$LOG_DIR/${CK}__${TS}.log"

  # Run the logged block in a SUBSHELL so `exit` only leaves this job's block
  # (not the whole script) and the per-job status line below always prints.
  (
    echo "================================================================"
    echo "[$(date)] ▶ ckpt=$CK  testset=$TESTSET"
    echo "ckpt_path = $CKPT_PATH"
    echo "config    = $CFG_PATH"
    echo "metalst   = $METALST"
    echo "gen_path  = $GEN_PATH"
    echo "================================================================"

    if [ ! -f "$CKPT_PATH" ]; then echo "⚠️  ckpt missing: $CKPT_PATH — skip"; echo "STATUS=missing_ckpt"; exit 0; fi
    if [ ! -f "$CFG_PATH" ];  then echo "⚠️  config missing: $CFG_PATH — skip"; echo "STATUS=missing_config"; exit 0; fi
    if [ ! -f "$METALST" ];   then echo "⚠️  manifest missing: $METALST — skip"; echo "STATUS=missing_manifest"; exit 0; fi
    EXPECT_WAVS=$(grep -cve '^[[:space:]]*$' "$METALST")

    # ---- 1) inference (8 GPU) ------------------------------------------------
    NWAV=$(ls "$GEN_PATH"/*.wav 2>/dev/null | wc -l)
    if [ "$NWAV" -ge "$EXPECT_WAVS" ]; then
      echo "inference: $NWAV/$EXPECT_WAVS wavs present — skip"
    else
      echo "---- inference (8 GPU): have $NWAV/$EXPECT_WAVS ----"
      accelerate launch --num_processes "$GPU_NUMS" --main_process_port "$MAIN_PORT" \
        "$INFER_PY" \
        -n "$CK" -t "$TESTSET" --seed "$SEED" \
        --ckpt_path "$CKPT_PATH" --config_path "$CFG_PATH" \
        -nfe "$NFE_STEP" -o "$ODE_METHOD" -ss "$SWAY_SAMPLING" \
        --cfg_scale "$CFG_STRENGTH" --sample_strategy "$SAMPLE_STRATEGY" \
        --tag "$TAG" \
        -p "$PROMPT_DIR" --metalst "$METALST" \
        --output_dir "$GEN_PATH"
      NWAV=$(ls "$GEN_PATH"/*.wav 2>/dev/null | wc -l)
      echo "inference done: $NWAV wavs"
    fi

    if [ "$NWAV" -lt "$EXPECT_WAVS" ]; then
      echo "⚠️  inference incomplete ($NWAV/$EXPECT_WAVS) — skip metrics (re-run to resume)"
      echo "STATUS=incomplete_infer"; exit 0
    fi

    # ---- 2) metrics: SIM -> UTMOS -> WER ------------------------------------
    if [ -f "$GEN_PATH/_sim_results.jsonl" ]; then echo "SIM: exists, skip"; else
      echo ">>> SIM ..."
      if [ "$TS" = "ls" ]; then
        python "$METRIC_PY" -e sim -l "$LANG" -p "$PROMPT_DIR" --gen_wav_dir "$GEN_PATH" --metalst "$METALST" --local --gpu_nums "$GPU_NUMS"
      else
        python "$METRIC_PY" -e sim -l "$LANG" --gen_wav_dir "$GEN_PATH" --metalst "$METALST" --local --gpu_nums "$GPU_NUMS"
      fi
    fi

    if [ -f "$GEN_PATH/_utmos_results.jsonl" ]; then echo "UTMOS: exists, skip"; else
      echo ">>> UTMOS ..."; python "$UTMOS_PY" --audio_dir "$GEN_PATH"; fi

    if [ -f "$GEN_PATH/_wer_results.jsonl" ]; then echo "WER: exists, skip"; else
      echo ">>> WER ($LANG) ..."
      if [ "$TS" = "ls" ]; then
        python "$METRIC_PY" -e wer -l "$LANG" -p "$PROMPT_DIR" --gen_wav_dir "$GEN_PATH" --metalst "$METALST" --local --gpu_nums "$GPU_NUMS"
      else
        python "$METRIC_PY" -e wer -l "$LANG" --gen_wav_dir "$GEN_PATH" --metalst "$METALST" --local --gpu_nums "$GPU_NUMS"
      fi
    fi

    echo "[$(date)] ✔ done ckpt=$CK testset=$TESTSET"
    echo "STATUS=done"
  ) > "$LOG" 2>&1

  local ST; ST=$(grep -o 'STATUS=[a-z_]*' "$LOG" | tail -1)
  echo "[$(date)] ckpt=$CK testset=$TS  ->  $LOG  ($ST)"
}

# ----------------------------------------------------------------------------
# Main loop — sequential over (checkpoint × testset).
# ----------------------------------------------------------------------------
echo "=============================================================="
echo " GROW single-machine 8-GPU eval"
echo "   checkpoints : $CKPTS"
echo "   test sets   : $TESTSETS"
echo "   GPUs        : $CUDA_VISIBLE_DEVICES  (num_processes=$GPU_NUMS)"
echo "   cfg=$CFG_STRENGTH ($SAMPLE_STRATEGY)  nfe=$NFE_STEP  seed=$SEED"
echo "   results -> $RESULTS_DIR"
echo "=============================================================="

for CK in $CKPTS; do
  for TS in $TESTSETS; do
    run_job "$CK" "$TS"
  done
done

# ----------------------------------------------------------------------------
# Summary — collect the final metric from every _*_results.jsonl produced.
# ----------------------------------------------------------------------------
echo ""
echo "=============================== SUMMARY ==============================="
echo "(WER/SIM are fractions: WER 0.02373 == paper 2.373 ; multiply WER by 100)"
printf "%-22s %-20s %8s %8s %8s\n" "checkpoint" "testset" "WER" "SIM" "UTMOS"
for CK in $CKPTS; do
  for TS in $TESTSETS; do
    case "$TS" in
      ls) TESTSET="ls_pc_test_clean" ;;
      en) TESTSET="seedtts_test_en" ;;
      zh) TESTSET="seedtts_test_zh" ;;
    esac
    GEN_PATH="$RESULTS_DIR/$CK/$TESTSET/seed${SEED}_${ODE_METHOD}_nfe${NFE_STEP}_${MEL_SPEC_TYPE}_ss${SWAY_SAMPLING}_cfg${CFG_STRENGTH}_bsz${INFER_BATCH_SIZE}_${TAG}_${TASK}"
    wer=$(grep -h "^WER:"   "$GEN_PATH/_wer_results.jsonl"   2>/dev/null | tail -1 | awk '{print $2}')
    sim=$(grep -h "^SIM:"   "$GEN_PATH/_sim_results.jsonl"   2>/dev/null | tail -1 | awk '{print $2}')
    utm=$(grep -h "^UTMOS:" "$GEN_PATH/_utmos_results.jsonl" 2>/dev/null | tail -1 | awk '{print $2}')
    printf "%-22s %-20s %8s %8s %8s\n" "$CK" "$TESTSET" "${wer:-–}" "${sim:-–}" "${utm:-–}"
  done
done
echo "======================================================================"
echo "[$(date)] ALL DONE. Per-job logs in $LOG_DIR/"

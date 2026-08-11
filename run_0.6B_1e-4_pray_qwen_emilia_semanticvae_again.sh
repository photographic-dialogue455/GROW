#!/usr/bin/env bash
# DiTAR pretraining — single machine, 8 GPU.
set -u

EXP_NAME="ditar_train_dit_emilia_0.6b_semanticvae_again"
EXP_DIR="ckpts/${EXP_NAME}"

GPUS_PER_NODE=8
CONFIG_FILE="general_bf16.yaml"

mkdir -p "$EXP_DIR"
echo "single-machine training -- GPUS_PER_NODE: ${GPUS_PER_NODE}"

# Hydra overrides. Point train/val at your prepared datasets (see the Data
# Preparation section of README.md — build them with prepare_emilia_v2.py then
# convert_raw_arrow_to_save_to_disk.py). Both must be in save_to_disk format.
hydra_args="
++hydra.run.dir=$EXP_DIR
++ckpts.save_dir=$EXP_DIR
++wandb_config.wandb_run_name=$EXP_NAME
++datasets.train_ds_path=data/Emilia_ZH_EN_char
++datasets.val_ds_path=data/LibriTTS_test-clean_char_1280
"

accelerate launch \
    --config_file "$CONFIG_FILE" \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --machine_rank 0 \
    src/DiTAR/train/train_ditar.py \
    -cn t1_ditar_0.6b_semanticvae \
    $hydra_args \
    2>&1 | tee "$EXP_DIR/train.log"

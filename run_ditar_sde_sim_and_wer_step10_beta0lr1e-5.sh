EXP_NAME="ditar_sde_sim_and_wer_step10_beta0lr1e-5"
EXP_DIR="ckpts/${EXP_NAME}"

# =========================================================
#  单机 8 卡配置
# =========================================================
GPUS_PER_NODE=8
CONFIG_FILE="general_bf16.yaml"

mkdir -p "$EXP_DIR"
echo "单机训练 -- GPUS_PER_NODE: ${GPUS_PER_NODE}"

# =========================================================
# Hydra 参数
# =========================================================
hydra_args="
++hydra.run.dir=${EXP_DIR}
++ckpts.save_dir=${EXP_DIR}
++ckpts.logger=wandb
++model.orw.n_rollout=8
++model.orw.beta=0
++model.orw.alpha=1.0
++model.orw.rm_method=grpo
++model.orw.sample_steps=10
++ckpts.save_per_updates=25
++ckpts.last_per_updates=25
++model.orw.reward_type=sim_and_wer
++model.orw.wer_weight=1.0
++model.orw.sim_weight=1.0
++optim.learning_rate=1e-5
"

# =========================================================
# 启动命令 (单机 8 卡)
# =========================================================
accelerate launch \
    --config_file "$CONFIG_FILE" \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --machine_rank 0 \
    src/DiTAR/train/train_ditar_sde.py \
    -cn t1_ditar_0.6b_sde \
    $hydra_args \
    2>&1 | tee "$EXP_DIR/train.log"




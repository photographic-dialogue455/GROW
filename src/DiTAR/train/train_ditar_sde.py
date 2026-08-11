import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf
from DiTAR.model import DiTAR_ORW_Trainer, DiTAR_SDE
from DiTAR.model.ldm_dataset import load_ldm_dataset
import logging

logger = logging.getLogger(__name__)

os.chdir(str(files("DiTAR").joinpath("../..")))


def create_model(model_cfg, tokenizer):
    return DiTAR_SDE(ditar_cfg=model_cfg, tokenizer=tokenizer)


@hydra.main(version_base="1.3", config_path=str(files("DiTAR").joinpath("configs")), config_name=None)
def main(cfg):
    logger.info(f"cfg: \n{cfg}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone.pretrained_LM_path)

    model = create_model(cfg.model, tokenizer)

    trainer = DiTAR_ORW_Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("DiTAR").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_config=cfg.wandb_config,
        last_per_updates=cfg.ckpts.last_per_updates,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        model_cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        loss_weight=OmegaConf.to_container(cfg.loss_weight, resolve=True),
        checkpoint_step=cfg.ckpts.checkpoint_step,
        max_updates=cfg.ckpts.get("max_updates", None),
        early_stop_min_acc=cfg.ckpts.get("early_stop_min_acc", None),
        early_stop_reward_types=cfg.ckpts.get("early_stop_reward_types", ("sim_and_wer",)),
    )

    train_dataset = load_ldm_dataset(
        cfg.datasets.train_ds_path,
        audio_type=cfg.model.audio_type,
        target_sample_rate=cfg.datasets.target_sample_rate,
    )
    val_dataset = None  # SDE/Flow-GRPO runs no validation forward (rollout is too costly)

    trainer.train(
        train_dataset,
        val_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()

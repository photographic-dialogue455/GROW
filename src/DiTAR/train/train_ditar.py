# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from DiTAR.model import DiTARTrainer, DiTAR
from DiTAR.model.ldm_dataset import load_ldm_dataset
import logging

logger = logging.getLogger(__name__)

os.chdir(str(files("DiTAR").joinpath("../..")))  # change working directory to root of project (local editable)

def create_model(model_cfg, tokenizer):
    if model_cfg.name=="DiTAR":
        model = DiTAR(
            ditar_cfg = model_cfg,
            tokenizer = tokenizer,
        )
    else:
        raise ValueError(f"Unknown model class: {model_cfg.name}")

    return model

@hydra.main(version_base="1.3", config_path=str(files("DiTAR").joinpath("configs")), config_name=None)
def main(cfg):
    logger.info(f"cfg: \n{cfg}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone.pretrained_LM_path)

    model = create_model(cfg.model, tokenizer)

    # init trainer
    trainer = DiTARTrainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("DiTAR").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_config = cfg.wandb_config,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=cfg.ckpts.log_samples,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=cfg.model.vocoder.mel_spec_type, #cfg.model.audio_type,
        vocoder_path=cfg.model.vocoder.local_path,
        is_local=cfg.model.vocoder.is_local,
        local_path=cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        loss_weight=cfg.loss_weight,
        checkpoint_step=cfg.ckpts.checkpoint_step,
    )

    train_dataset = load_ldm_dataset(
        cfg.datasets.train_ds_path,
        audio_type=cfg.model.audio_type,
        target_sample_rate = cfg.datasets.target_sample_rate
    )
    val_dataset = load_ldm_dataset(
        cfg.datasets.val_ds_path,
        audio_type=cfg.model.audio_type,
        target_sample_rate = cfg.datasets.target_sample_rate
    )

    trainer.train(
        train_dataset,
        val_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
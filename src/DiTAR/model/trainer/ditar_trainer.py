"""
DiTAR trainer.

Two losses, both weighted via ``self.loss_weight`` (from the training YAML):
  - diff_loss  (flow-matching diffusion loss)
  - stop_loss  (EOS / patch-stop classifier loss)

Total loss = sum(self.loss_weight[k] * output[k]) over keys present in output.
"""
from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from DiTAR.model.dataset import DynamicBatchSampler
from DiTAR.model.utils import default, exists, count_parameters, is_debugging

local_logger = get_logger(__name__)


class DiTARTrainer:
    def __init__(
        self,
        model,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger: str | None = "wandb",
        wandb_config: dict = dict(),
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",
        vocoder_path: str = "",
        is_local: bool = False,
        local_path: str = "",
        model_cfg_dict: dict = dict(),
        loss_weight: dict = dict(),
        n_save_samples: int = 32,
        checkpoint_step=None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            wandb_init_kwargs = {
                "resume": "allow",
                "name": wandb_config.wandb_run_name,
                "mode": wandb_config.mode,
            }
            if is_debugging():
                wandb_init_kwargs["mode"] = "disabled"

            if exists(wandb_config.resume_id):
                wandb_init_kwargs |= {"id": wandb_config.resume_id}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_config.project,
                init_kwargs={"wandb": wandb_init_kwargs},
                config=model_cfg_dict,
            )
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                wandb.define_metric("update")
                wandb.define_metric("train/*", step_metric="update")
                wandb.define_metric("val/*", step_metric="update")

            self.accelerator.wait_for_everyone()

        self.model = model

        if self.is_main:
            generator = getattr(model, "generator", None)
            if generator is not None:
                # Detach so deepcopy skips the un-copyable vocoder submodule.
                del model.generator
            try:
                self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            finally:
                if generator is not None:
                    # Re-attach to both models; they share the same frozen vocoder.
                    model.generator = generator
                    self.ema_model.ema_model.generator = generator
            self.ema_model.to(self.accelerator.device)

            local_logger.info(f"{model}")
            local_logger.info(f"Using logger: {logger}")
            local_logger.info(f"Total trainable parameters: {count_parameters(model)} M")
            local_logger.info(f"CausalAR parameters: {count_parameters(model.causalAR)} M")
            local_logger.info(f"LocDiT parameters: {count_parameters(model.LocDiT)} M")
            if grad_accumulation_steps > 1:
                local_logger.info(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.mel_spec_type = mel_spec_type
        self.is_local = is_local
        self.local_path = local_path
        self.loss_weight = loss_weight

        if bnb_optimizer:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.checkpoint_step = checkpoint_step

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            try:
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict()
            except Exception:
                optimizer_state_dict = self.optimizer.state_dict()
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=optimizer_state_dict,
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                local_logger.info(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        local_logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if self.checkpoint_step is not None:
            latest_checkpoint = self.checkpoint_step
        elif "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        if "update" in checkpoint or "step" in checkpoint:
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    local_logger.info(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]            
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    # total loss = sum(self.loss_weight[k] * output[k]) over present keys
    def _compose_loss(self, output):
        components = []
        for k, v in self.loss_weight.items():
            if k in output and output[k] is not None:
                components.append(v * output[k])
        return sum(components)

    def train(self, train_dataset: Dataset, val_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        from DiTAR.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef
        device = self.accelerator.device

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )

            val_sampler = SequentialSampler(val_dataset)
            val_batch_sampler = DynamicBatchSampler(
                val_sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            val_dataloader = DataLoader(
                val_dataset,
                collate_fn=val_dataset.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=val_batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        warmup_updates = (self.num_warmup_updates * self.accelerator.num_processes)
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        val_dataloader = self.accelerator.prepare(val_dataloader)
        start_update = self.load_checkpoint()
        global_update = start_update

        if self.log_samples:
            vocoder = load_vocoder(
                vocoder_name=self.mel_spec_type,
                is_local=self.is_local,
                local_path=self.local_path,
            )
            vocoder = vocoder.to(self.accelerator.device)
            target_sample_rate = 16000
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            # Accumulators track sample-weighted sums (Σ loss_i * bsz_i); the
            # reported metric is Σ(loss*bsz)/Σbsz — a true per-sample mean,
            # robust to dynamic batch sizes and grad_accumulation_steps > 1.
            train_loss_sum = 0.0
            train_diff_loss_sum = 0.0
            train_stop_loss_sum = 0.0
            train_stop_acc_sum = 0.0
            train_stop_sentence_acc_sum = 0.0
            train_n_samples = 0

            interval_loss_sum = 0.0
            interval_diff_loss_sum = 0.0
            interval_stop_loss_sum = 0.0
            interval_stop_acc_sum = 0.0
            interval_stop_sentence_acc_sum = 0.0
            interval_n_samples = 0

            for i_batch, batch in enumerate(current_dataloader):
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    text_lengths = batch["text_lengths"]
                    text_mask = batch["text_mask"]

                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_mask = batch["mel_mask"]
                    bsz = len(batch["text"])

                    output = self.model(
                        raw_wav=mel_spec,
                        raw_wav_mask=mel_mask,
                        text=text_inputs,
                        text_lengths=text_lengths,
                        text_padding_mask=text_mask,
                    )

                    loss = self._compose_loss(output)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                logs = {
                    "train/loss": loss.item(),
                    "train/diff_loss": output["diff_loss"].item(),
                    "train/stop_loss": output["stop_loss"].item(),
                    "train/stop_acc": output["stop_acc"].item(),
                    "train/stop_sentence_acc": output["stop_sentence_acc"].item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "update": global_update,
                }

                # Sample-weighted accumulation: each batch contributes
                # `loss * bsz` so dividing by Σbsz yields the per-sample mean.
                train_loss_sum += loss.item() * bsz
                train_diff_loss_sum += output["diff_loss"].item() * bsz
                train_stop_loss_sum += output["stop_loss"].item() * bsz
                train_stop_acc_sum += output["stop_acc"].item() * bsz
                train_stop_sentence_acc_sum += output["stop_sentence_acc"].item() * bsz
                train_n_samples += bsz

                interval_loss_sum += loss.item() * bsz
                interval_diff_loss_sum += output["diff_loss"].item() * bsz
                interval_stop_loss_sum += output["stop_loss"].item() * bsz
                interval_stop_acc_sum += output["stop_acc"].item() * bsz
                interval_stop_sentence_acc_sum += output["stop_sentence_acc"].item() * bsz
                interval_n_samples += bsz

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(logs, step=global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_mask[0].sum().item()
                        infer_text = [text_inputs[0] + text_inputs[0]]
                        with torch.inference_mode():
                            if self.mel_spec_type == "semanticvae":
                                generated, prompt_vae = self.accelerator.unwrap_model(self.model).sample(
                                    prompt_wav=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    prompt_wav_mask=mel_mask[0:1, :ref_audio_len],
                                    text=infer_text,
                                    text_lengths=text_lengths[0] * 2,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                )

                            generated = generated.to(torch.float32)
                            if self.mel_spec_type == "semanticvae":
                                gen_audio = vocoder.decode(generated).squeeze(0).cpu()
                                reconstruct_audio = vocoder.decode(prompt_vae).squeeze(0).cpu()
                            ref_audio = mel_spec[0:1].squeeze(-1).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_recon.wav", reconstruct_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )

                    # Sample-weighted mean over the interval (max(...,1) guards
                    # the rare all-skipped case).
                    interval_n = max(interval_n_samples, 1)
                    avg_loss = interval_loss_sum / interval_n
                    avg_diff_loss = interval_diff_loss_sum / interval_n
                    avg_stop_loss = interval_stop_loss_sum / interval_n
                    avg_stop_acc = interval_stop_acc_sum / interval_n
                    avg_stop_sentence_acc = interval_stop_sentence_acc_sum / interval_n

                    if self.accelerator.is_local_main_process:
                        local_logger.info(
                            f"save checkpoint at update {global_update}, "
                            f"loss: {avg_loss}, "
                            f"diff_loss: {avg_diff_loss}, "
                            f"stop_loss: {avg_stop_loss}, "
                            f"stop_acc: {avg_stop_acc}, "
                            f"stop_sentence_acc: {avg_stop_sentence_acc}, "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

                        interval_logs = {
                            "interval/loss": avg_loss,
                            "interval/diff_loss": avg_diff_loss,
                            "interval/stop_loss": avg_stop_loss,
                            "interval/stop_acc": avg_stop_acc,
                            "interval/stop_sentence_acc": avg_stop_sentence_acc,
                            "interval/lr": self.scheduler.get_last_lr()[0],
                            "interval/update": global_update,
                        }
                        self.accelerator.log(interval_logs, step=global_update)

                    interval_loss_sum = 0.0
                    interval_diff_loss_sum = 0.0
                    interval_stop_loss_sum = 0.0
                    interval_stop_acc_sum = 0.0
                    interval_stop_sentence_acc_sum = 0.0
                    interval_n_samples = 0

                    # =========  evaluation  =========
                    val_progress_bar = tqdm(
                        range(math.ceil(len(val_dataloader) / self.grad_accumulation_steps)),
                        desc=f"Epoch {epoch + 1}/{self.epochs}",
                        unit="update",
                        disable=not self.accelerator.is_local_main_process,
                        initial=0,
                    )

                    self.model.eval()

                    # 1-d tensors so gather_for_metrics behaves consistently
                    # across accelerate versions (0-d gathering is ill-specified).
                    eval_device = self.accelerator.device
                    eval_loss_sum = torch.zeros(1, device=eval_device)
                    eval_diff_loss_sum = torch.zeros(1, device=eval_device)
                    eval_stop_loss_sum = torch.zeros(1, device=eval_device)
                    eval_stop_acc_sum = torch.zeros(1, device=eval_device)
                    eval_stop_sentence_acc_sum = torch.zeros(1, device=eval_device)
                    n_samples = 0

                    for i, batch in enumerate(val_dataloader):
                        if i >= 3 and is_debugging():
                            break
                        with self.accelerator.accumulate(self.model):
                            text_inputs = batch["text"]
                            text_lengths = batch["text_lengths"]
                            text_mask = batch["text_mask"]

                            mel_spec = batch["mel"].permute(0, 2, 1)
                            mel_mask = batch["mel_mask"]
                            bsz = len(batch['text'])

                            with torch.no_grad():
                                val_output = self.model(
                                    raw_wav=mel_spec,
                                    raw_wav_mask=mel_mask,
                                    text=text_inputs,
                                    text_lengths=text_lengths,
                                    text_padding_mask=text_mask,
                                )
                                val_loss = self._compose_loss(val_output)

                            # Sample-weighted: contributes (per-batch mean) * bsz,
                            # so gathered_sum / Σbsz is the per-sample val mean.
                            eval_loss_sum              += val_loss.detach().reshape(1)                         * bsz
                            eval_diff_loss_sum         += val_output["diff_loss"].detach().reshape(1)          * bsz
                            eval_stop_loss_sum         += val_output["stop_loss"].detach().reshape(1)          * bsz
                            eval_stop_acc_sum          += val_output["stop_acc"].detach().reshape(1)           * bsz
                            eval_stop_sentence_acc_sum += val_output["stop_sentence_acc"].detach().reshape(1)  * bsz
                            n_samples += bsz

                            val_logs = {
                                "val/loss": val_loss.item(),
                                "val/diff_loss": val_output["diff_loss"].item(),
                                "val/stop_loss": val_output["stop_loss"].item(),
                                "val/stop_acc": val_output["stop_acc"].item(),
                                "val/stop_sentence_acc": val_output["stop_sentence_acc"].item(),
                                "val/lr": self.scheduler.get_last_lr()[0],
                                "update": global_update,
                            }

                            val_progress_bar.update(1)
                            val_progress_bar.set_postfix(**val_logs)

                            if self.accelerator.is_local_main_process:
                                self.accelerator.log(val_logs, step=global_update)

                    n_samples_tensor = torch.tensor([float(n_samples)], device=eval_device)
                    (
                        eval_loss_sum,
                        eval_diff_loss_sum,
                        eval_stop_loss_sum,
                        eval_stop_acc_sum,
                        eval_stop_sentence_acc_sum,
                        n_samples_tensor,
                    ) = self.accelerator.gather_for_metrics(
                        (
                            eval_loss_sum,
                            eval_diff_loss_sum,
                            eval_stop_loss_sum,
                            eval_stop_acc_sum,
                            eval_stop_sentence_acc_sum,
                            n_samples_tensor,
                        )
                    )
                    # clamp(min=1.0) guards the rare case of zero val samples on a process.
                    total_n = n_samples_tensor.sum().clamp(min=1.0)
                    epoch_eval_loss              = (eval_loss_sum.sum()              / total_n).item()
                    epoch_eval_diff_loss         = (eval_diff_loss_sum.sum()         / total_n).item()
                    epoch_eval_stop_loss         = (eval_stop_loss_sum.sum()         / total_n).item()
                    epoch_eval_stop_acc          = (eval_stop_acc_sum.sum()          / total_n).item()
                    epoch_eval_stop_sentence_acc = (eval_stop_sentence_acc_sum.sum() / total_n).item()

                    self.model.train()
                    if self.accelerator.is_local_main_process:
                        local_logger.info(
                            f"validation result at update {global_update}, "
                            f"loss: {epoch_eval_loss}, "
                            f"diff_loss: {epoch_eval_diff_loss}, "
                            f"stop_loss: {epoch_eval_stop_loss}, "
                            f"stop_acc: {epoch_eval_stop_acc}, "
                            f"stop_sentence_acc: {epoch_eval_stop_sentence_acc}, "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

            # Sample-weighted epoch averages for train metrics.
            epoch_n = max(train_n_samples, 1)
            epoch_train_loss = train_loss_sum / epoch_n
            epoch_train_diff_loss = train_diff_loss_sum / epoch_n
            epoch_train_stop_loss = train_stop_loss_sum / epoch_n
            epoch_train_stop_acc = train_stop_acc_sum / epoch_n
            epoch_train_stop_sentence_acc = train_stop_sentence_acc_sum / epoch_n

            if self.accelerator.is_local_main_process:
                epoch_logs = {
                    "epoch/loss": epoch_train_loss,
                    "epoch/diff_loss": epoch_train_diff_loss,
                    "epoch/stop_loss": epoch_train_stop_loss,
                    "epoch/stop_acc": epoch_train_stop_acc,
                    "epoch/stop_sentence_acc": epoch_train_stop_sentence_acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                self.accelerator.log(epoch_logs, step=global_update)
                local_logger.info(
                    f"Epoch {epoch} completed. "
                    f"Loss: {epoch_train_loss}, "
                    f"Diff Loss: {epoch_train_diff_loss}, "
                    f"Stop Loss: {epoch_train_stop_loss}, "
                    f"stop_acc: {epoch_train_stop_acc}, "
                    f"stop_sentence_acc: {epoch_train_stop_sentence_acc}, "
                    f"lr: {self.scheduler.get_last_lr()[0]}"
                )

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()

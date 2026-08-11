from __future__ import annotations

import gc
import math
import os
from datetime import timedelta

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR  # ORW uses a constant LR (no warmup/decay)
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from DiTAR.model.dataset import DynamicBatchSampler
from DiTAR.model.utils import default, exists, count_parameters, is_debugging

local_logger = get_logger(__name__)


class DiTAR_ORW_Trainer:
    def __init__(
        self,
        model,
        epochs,
        learning_rate,
        save_per_updates=100,
        keep_last_n_checkpoints: int = -1,
        checkpoint_path=None,
        batch_size_per_gpu=1,
        batch_size_type: str = "sample",
        max_samples=1,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger: str | None = "wandb",
        wandb_config: dict = dict(),
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        model_cfg_dict: dict = dict(),
        loss_weight: dict = dict(),
        checkpoint_step=None,
        max_updates=None,
        early_stop_min_acc=None,
        early_stop_reward_types=("sim_and_wer",),
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)

        # A rank that repeatedly draws r_std==0 batches keeps re-rolling and
        # reaches the backward all-reduce late; other ranks wait there. This is
        # straggler wait, not desync — so just relax the NCCL collective timeout
        # (default 10min is tight for re-rolls).
        nccl_timeout_min = int(os.environ.get("ORW_NCCL_TIMEOUT_MIN", "60"))
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=nccl_timeout_min))

        if logger == "wandb" and not wandb.api.api_key:
            logger = None

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs, timeout_kwargs],
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
                wandb.define_metric("orw/*", step_metric="update")
            self.accelerator.wait_for_everyone()

        self.model = model

        if self.is_main:
            local_logger.info(f"Using logger: {logger}")
            local_logger.info(f"Total trainable parameters: {count_parameters(model)} M")
            local_logger.info(f"CausalAR parameters: {count_parameters(model.causalAR)} M")
            local_logger.info(f"LocDiT parameters: {count_parameters(model.LocDiT)} M")
            local_logger.info(f"Policy parameters: {count_parameters(model.policy)} M")

        self.epochs = epochs
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_ditar_orw")
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.loss_weight = loss_weight

        # Freeze the generator before building the optimiser.
        for name, param in model.named_parameters():
            if "generator" in name:
                param.requires_grad = False

        # Optimiser sees only trainable params (the policy, plus anything left
        # as requires_grad=True).
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if bnb_optimizer:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate)
        else:
            self.optimizer = AdamW(trainable_params, lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.checkpoint_step = checkpoint_step
        # Optional: stop after this many global_updates (None = unlimited).
        self.max_updates = max_updates

        self.early_stop_min_acc = early_stop_min_acc
        self.early_stop_reward_types = tuple(early_stop_reward_types or ())
        # reward_type read from the unwrapped model; None -> matches no gate type.
        self._reward_type = getattr(self.accelerator.unwrap_model(self.model), "reward_type", None)
        self.early_stop_enabled = (
            self.early_stop_min_acc is not None
            and self._reward_type in self.early_stop_reward_types
        )
        if self.is_main and self.early_stop_min_acc is not None:
            if self.early_stop_enabled:
                local_logger.info(
                    f"[ORW] early-stop enabled: will stop if global implicit_acc < "
                    f"{self.early_stop_min_acc} (reward_type={self._reward_type})"
                )
            else:
                local_logger.info(
                    f"[ORW] early-stop configured (min_acc={self.early_stop_min_acc}) but inactive: "
                    f"reward_type={self._reward_type} not in {self.early_stop_reward_types}"
                )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # Checkpointing — save *policy only*. The reference checkpoint is never
    # overwritten; the infer copy is always re-derived from the policy.
    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            try:
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict()
            except Exception:
                optimizer_state_dict = self.optimizer.state_dict()
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).policy.state_dict(),
                optimizer_state_dict=optimizer_state_dict,
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            os.makedirs(self.checkpoint_path, exist_ok=True)
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

        ckpt = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        # Permissive about old/new layouts.
        if "ema_model_state_dict" in ckpt:
            sd = {
                k.replace("ema_model.", ""): v
                for k, v in ckpt["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt

        unwrapped = self.accelerator.unwrap_model(self.model)
        target = unwrapped.policy if hasattr(unwrapped, "policy") else unwrapped
        # Strip 'policy.' prefix if present.
        if any(k.startswith("policy.") for k in sd.keys()):
            sd = {k[len("policy."):]: v for k, v in sd.items() if k.startswith("policy.")}
        info = target.load_state_dict(sd, strict=False)
        if self.is_main:
            local_logger.info(
                f"[ORW] resume from {latest_checkpoint}: missing={len(info.missing_keys)}, "
                f"unexpected={len(info.unexpected_keys)}"
            )
        # Restore optimizer/scheduler only when present (a plain SFT checkpoint
        # has neither — ORW normally starts from a fresh AdamW).
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                local_logger.warning(f"[ORW] optimizer state restore failed: {e}")
        if "scheduler_state_dict" in ckpt and getattr(self, "scheduler", None) is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                local_logger.warning(f"[ORW] scheduler state restore failed: {e}")

        update = ckpt.get("update", ckpt.get("step", 0))
        if "step" in ckpt and "update" not in ckpt:
            update = update // max(1, self.grad_accumulation_steps)

        del ckpt
        gc.collect()
        return update

    # total loss = sum(loss_weight[k] * output[k]) over present keys
    def _compose_loss(self, output):
        components = []
        for k, v in self.loss_weight.items():
            if k in output and output[k] is not None:
                components.append(v * output[k])
        return sum(components)

    def train(self, train_dataset: Dataset, val_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
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
        else:
            raise ValueError(f"Unsupported batch_size_type: {self.batch_size_type}")

        # Constant LR for the ORW stage (factor fixed at 1.0, no warmup/decay).
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _step: 1.0)
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_update = self.load_checkpoint()
        global_update = start_update

        # Mid-epoch resume: skip the dataloader ahead so already-processed
        # batches aren't replayed. Mirrors DiTARTrainer.train(...).
        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step) if orig_epoch_step > 0 else 0
            skipped_batch = start_step % orig_epoch_step if orig_epoch_step > 0 else 0
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0
            skipped_batch = 0
            skipped_dataloader = None

        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "init_reward_models"):
            unwrapped.init_reward_models(self.accelerator.device)

        # Early-stop flag; only depends on global_update / sync_gradients (equal
        # across ranks), so all ranks break at the same i_batch — no desync.
        stop_training = False
        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            # Keep ref/infer in eval mode regardless of model.train().
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "ref"):
                unwrapped.ref.eval()
            if hasattr(unwrapped, "infer"):
                unwrapped.infer.eval()

            # On the first (resumed) epoch use the skipped dataloader; later
            # epochs use the full dataloader from the start.
            if exists(resumable_with_seed) and epoch == skipped_epoch and skipped_dataloader is not None:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            # Infinite iterator + step count: ORW re-rolls invalid batches, so
            # we loop over a step budget rather than over the dataloader.
            def get_infinite_iterator(dataloader):
                while True:
                    for b in dataloader:
                        yield b

            infinite_data_iter = get_infinite_iterator(current_dataloader)
            epoch_steps = math.ceil(len(current_dataloader) / self.grad_accumulation_steps) * self.grad_accumulation_steps
            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for i_batch in range(epoch_steps):
                with self.accelerator.accumulate(self.model):
                    is_valid_batch = False
                    # Re-roll until we draw a batch with std != 0 (a zero-std
                    # batch gives no usable reward signal).
                    while not is_valid_batch:
                        batch = next(infinite_data_iter)
                        text_inputs = batch["text"]
                        text_lengths = batch["text_lengths"]
                        text_mask = batch["text_mask"]
                        mel_spec = batch["mel"].permute(0, 2, 1)   # (B, T, 1)
                        mel_mask = batch["mel_mask"]
                        gen_text = batch["gen_text"]
                        lang = batch.get("lang", None)

                        output = self.model(
                            raw_wav=mel_spec,
                            raw_wav_mask=mel_mask,
                            text=text_inputs,
                            text_lengths=text_lengths,
                            text_padding_mask=text_mask,
                            gen_text = gen_text,
                            lang = lang,
                        )
                        if output["r_std"].item() != 0:
                            is_valid_batch = True

                    loss = self._compose_loss(output)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                self.max_grad_norm,
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_update += 1
                    progress_bar.update(1)

                    # Cross-rank aggregation: collective ops — every rank MUST
                    # run these, so keep them OUT of any is_main branch or it
                    # deadlocks. Done once per optimiser step (sync_gradients).
                    # all-reduce(mean) is an unweighted per-GPU average — fine as
                    # monitoring, but not a sample-weighted global mean; the
                    # aggregated r_std is the mean of per-GPU stds, not a true
                    # global std.
                    local_vals = torch.stack([
                        loss.detach(),
                        output["diff_loss"].detach(),
                        output["kl_loss"].detach(),
                        output["pred_loss"].detach(),
                        output["implicit_acc"].detach(),
                        output["r_std"].detach(),
                    ]).float()
                    g = self.accelerator.reduce(local_vals, reduction="mean").tolist()
                    reward_local = output["reward_per_rollout"].reshape(-1).float()   # (n_rollout,)
                    weights_local = output["weights_per_rollout"].reshape(-1).float() # (n_rollout,)
                    reward_all = self.accelerator.gather(reward_local)               # (num_proc*n_rollout,)
                    weights_all = self.accelerator.gather(weights_local)             # (num_proc*n_rollout,)

                    logs = {
                        # —— per-rank (rank0 local; single-GPU jitter / debugging) ——
                        "train/loss": loss.item(),
                        "train/diff_loss": output["diff_loss"].item(),
                        "orw/kl_loss": output["kl_loss"].item(),
                        "orw/pred_loss": output["pred_loss"].item(),
                        "orw/implicit_acc": output["implicit_acc"].item(),
                        "orw/r_std": output["r_std"].item(),
                        # —— global (all-GPU average; real optimisation trend) ——
                        "train/loss_global": g[0],
                        "train/diff_loss_global": g[1],
                        "orw/kl_loss_global": g[2],
                        "orw/pred_loss_global": g[3],
                        "orw/implicit_acc_global": g[4],
                        "orw/r_std_global": g[5],
                        # —— global per-rollout weight/reward stats (min/max/mean) ——
                        "orw/weights_min_global": weights_all.min().item(),
                        "orw/weights_max_global": weights_all.max().item(),
                        "orw/weights_mean_global": weights_all.mean().item(),
                        "orw/weights_absmean_global": weights_all.abs().mean().item(),
                        "orw/reward_min_global": reward_all.min().item(),
                        "orw/reward_max_global": reward_all.max().item(),
                        "orw/reward_mean_global": reward_all.mean().item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "update": global_update,
                    }
                    progress_bar.set_postfix(**logs)

                    # Emit a logging.* record on every optimiser step: (1) when
                    # logger isn't "wandb" accelerator.log no-ops (no tracker),
                    # and (2) tqdm postfix goes to stderr, which Hydra's file
                    # handler doesn't capture.
                    if self.accelerator.is_local_main_process:
                        self.accelerator.log(logs, step=global_update)
                    if self.is_main:
                        local_logger.info(
                            f"step {global_update} | "
                            f"loss={logs['train/loss']:.4f}(g={logs['train/loss_global']:.4f}) "
                            f"diff={logs['train/diff_loss']:.4f}(g={logs['train/diff_loss_global']:.4f}) "
                            f"kl={logs['orw/kl_loss_global']:.4f} "
                            f"pred={logs['orw/pred_loss_global']:.4f} "
                            f"acc={logs['orw/implicit_acc_global']:.4f} "
                            f"r_std={logs['orw/r_std_global']:.4f} "
                            f"lr={logs['train/lr']:.2e}"
                        )
                        # rank0: print this step's per-rollout rewards/weights and
                        # the global weight stats — to spot weights degenerating
                        # to all-positive / near-uniform.
                        reward_r0 = reward_local.tolist()
                        weights_r0 = weights_local.tolist()
                        local_logger.info(
                            f"step {global_update} | rank0 "
                            f"reward={['%.4f' % v for v in reward_r0]} "
                            f"weights={['%.4f' % v for v in weights_r0]} "
                            f"| global w[min/max/mean/|mean|]="
                            f"{logs['orw/weights_min_global']:.4f}/"
                            f"{logs['orw/weights_max_global']:.4f}/"
                            f"{logs['orw/weights_mean_global']:.4f}/"
                            f"{logs['orw/weights_absmean_global']:.4f} "
                            f"r[min/max/mean]="
                            f"{logs['orw/reward_min_global']:.4f}/"
                            f"{logs['orw/reward_max_global']:.4f}/"
                            f"{logs['orw/reward_mean_global']:.4f}"
                        )

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)
                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                # Stop after max_updates (ckpt already saved above). Same
                # sync_gradients guard, so all ranks break together.
                if (
                    self.max_updates is not None
                    and self.accelerator.sync_gradients
                    and global_update >= self.max_updates
                ):
                    if self.is_main:
                        local_logger.info(
                            f"Reached max_updates={self.max_updates} at update {global_update}, stopping training."
                        )
                    stop_training = True
                    break

                if (
                    self.early_stop_enabled
                    and self.accelerator.sync_gradients
                    and logs["orw/implicit_acc_global"] < self.early_stop_min_acc
                ):
                    if self.is_main:
                        local_logger.info(
                            f"Early stop at update {global_update}: global implicit_acc="
                            f"{logs['orw/implicit_acc_global']:.4f} < {self.early_stop_min_acc} "
                            f"(reward_type={self._reward_type}), stopping training."
                        )
                    stop_training = True
                    break

            if stop_training:
                break

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()
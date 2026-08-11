"""DiTAR_ORW (GROW): advantage-weighted flow-matching with W2 reference anchor.

Per step: rollout n_rollout candidates from the frozen-this-step policy, reward
each (1-WER and/or WavLM sim per ``orw.reward_type``), turn rewards into weights
via ``rm_method``, then run policy + frozen ref on a shared (t, x0) and compute
``β·‖f_θ - f_ref‖² + Σ w_i·‖f_θ - target_i‖²``.

"""

from __future__ import annotations

import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiTAR.model.ditar.dit_ar import DiTAR
from DiTAR.model.utils import make_pad_mask
from DiTAR.model.local_asr import build_asr_model, run_asr_local_batch
from DiTAR.model.local_spk import build_wavlm_spk_model, run_wavlm_sim_local_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _no_grad_(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    return module

class DiTAR_ORW(nn.Module):
    """Three-headed DiTAR for online reward-weighted fine-tuning.
    ``self.policy``  — trainable copy (gets the gradient).
    ``self.ref``     — frozen reference, loaded from ``orw.ref_ckpt_path``.
    ``self.infer``   — eval copy used for rollouts; refreshed from ``policy`` each step.
    """

    def __init__(self, ditar_cfg, tokenizer):
        super().__init__()
        self.ditar_cfg = ditar_cfg
        orw_cfg = ditar_cfg.get("orw", None)
        self.orw_cfg = orw_cfg

        # ---- algorithm hyperparams ----
        self.beta = float(orw_cfg.get("beta", 0.0))                # KL/Wasserstein anchor weight
        self.alpha = float(orw_cfg.get("alpha", 0.1))              # temperature tau (rm_method=exp/exp_softmax)
        self.n_rollout = int(orw_cfg.get("n_rollout", 2))          # group size G
        self.rm_method = str(orw_cfg.get("rm_method", "grpo"))     # grpo | exp | exp_softmax
        self.ref_ckpt_path = str(orw_cfg.get("ref_ckpt_path", ""))

        # ---- rollout (sampling) hyperparams ----
        self.sample_steps = int(orw_cfg.get("sample_steps", 32))
        self.cfg_strength = float(orw_cfg.get("cfg_strength", 1.5))
        self.sway_sampling_coef = orw_cfg.get("sway_sampling_coef", -1.0)
        self.max_seq_length = int(orw_cfg.get("max_seq_length", 155)) # ~15s
        self.min_seq_length = int(orw_cfg.get("min_seq_length", 10))  # ~1.5s buffer

        # loss velocity: conditional vc (False, default) vs CFG-guided vg (True)。
        self.loss_cfg = bool(orw_cfg.get("loss_cfg", False))

        # ---- reward: "wer" (1-WER) / "sim" (WavLM sim) / "sim_and_wer" (both) ----
        self.reward_type = str(orw_cfg.get("reward_type", "wer")).lower()

        # "sim_and_wer": z-score-normalize each reward within its group, then add with these weights (equal by default).
        self.sim_weight = float(orw_cfg.get("sim_weight", 1.0))
        self.wer_weight = float(orw_cfg.get("wer_weight", 1.0))

        # ---- speaker-similarity (WavLM ECAPA-TDNN) hyperparams ----
        spk_cfg = orw_cfg.get("spk", {})
        self.spk_ckpt_path = str(spk_cfg.get("ckpt_path", "") or "checkpoints/wavlm_large_finetune.pth")
        # gen output and prompt are both 16 kHz, so no resampling before WavLM by default.
        self.spk_gen_sr = int(spk_cfg.get("gen_sample_rate", 16000))
        self.spk_prompt_sr = int(spk_cfg.get("prompt_sample_rate", 16000))

        # ---- reward model handles ----
        self.asr_zh = None
        self.asr_en = None
        self.spk_model = None

        # ---- expose patch_size for downstream padding ----
        self.patch_size = ditar_cfg.patch_size

        # ---- build the three sub-DiTARs ----
        self.policy = DiTAR(ditar_cfg=ditar_cfg, tokenizer=tokenizer)
        self.ref = DiTAR(ditar_cfg=ditar_cfg, tokenizer=tokenizer)
        self.infer = DiTAR(ditar_cfg=ditar_cfg, tokenizer=tokenizer)

        # Share one frozen BigVGAN-VAE vocoder across all three (saves ~2× VRAM).
        if hasattr(self.policy, "generator") and self.policy.generator is not None:
            shared_gen = self.policy.generator
            # detach old refs *before* re-wiring so they get freed immediately
            old = self.ref.generator
            self.ref.generator = shared_gen
            del old
            old = self.infer.generator
            self.infer.generator = shared_gen
            del old

        # ---- load ref checkpoint into ref AND policy so at step 0 they are
        # bit-identical and KL(policy || ref) == 0. Read the ckpt once (torch.load
        # is the cost), then apply to both. Trainer overwrites policy on resume.
        if self.ref_ckpt_path:
            sd = self._read_ckpt_state_dict(self.ref_ckpt_path)
            self._apply_state_dict(self.ref, sd, name="ref")
            self._apply_state_dict(self.policy, sd, name="policy")
            del sd
            torch.cuda.empty_cache()
        else:
            logger.warning(
                "model.orw.ref_ckpt_path is empty; ref model will start "
                "with random init. This is almost never what you want."
            )

        # ---- freeze ref + infer ----
        _no_grad_(self.ref)
        _no_grad_(self.infer)
        self.ref.eval()
        self.infer.eval()

    # ------------------------------------------------------------------
    # Pass-through delegates so the trainer sees the same surface as DiTAR.
    # ------------------------------------------------------------------
    @property
    def causalAR(self):
        return self.policy.causalAR

    @property
    def LocDiT(self):
        return self.policy.LocDiT

    @property
    def generator(self):
        return self.policy.generator

    @generator.setter
    def generator(self, v):
        self.policy.generator = v

    @property
    def mel_spec_type(self):
        return self.policy.mel_spec_type

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, *args, **kwargs):
        """Sampling delegates to the policy (used for log_samples)."""
        return self.policy.sample(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint loader: split into read (expensive torch.load, done ONCE)
    # and apply (cheap, per module) so loading into both ref and policy only
    # touches disk once. ``name`` is cosmetic (logging).
    # ------------------------------------------------------------------
    def _read_ckpt_state_dict(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ref_ckpt_path not found: {path}")
        logger.info(f"[DiTAR_ORW] Loading DiTAR weights from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # inference uses EMA weights by default; align with that here
        if "ema_model_state_dict" in ckpt:
            # Strip the "ema_model." prefix that EMA writes in DiTARTrainer.
            sd = {
                k.replace("ema_model.", ""): v
                for k, v in ckpt["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }

        # Strip a top-level "policy." prefix if the ckpt came from an ORW wrapper.
        if any(k.startswith("policy.") for k in sd.keys()):
            sd = {k[len("policy."):]: v for k, v in sd.items() if k.startswith("policy.")}

        del ckpt
        return sd

    def _apply_state_dict(self, module, sd: dict, name: str = "module"):
        info = module.load_state_dict(sd, strict=False)
        logger.info(
            f"[DiTAR_ORW] {name} load ok: {info}"
            f"unexpected={len(info.unexpected_keys)}, "
            f"missing_in_ckpt={len(info.missing_keys)}"
        )

    # ------------------------------------------------------------------
    # Reward-model init. Called ONCE from the trainer after accelerator.prepare
    # (so ``device`` is the real per-rank GPU). Keeps backends resident so the
    # forward pass never (re)loads a model mid-step.
    # ------------------------------------------------------------------
    def init_reward_models(self, device):
        need_sim = self.reward_type in ("sim", "sim_and_wer")
        need_wer = self.reward_type in ("wer", "sim_and_wer")
        if need_sim:
            if self.spk_model is None:
                logger.info("[DiTAR_ORW] building WavLM speaker encoder (sim reward)")
                self.spk_model = build_wavlm_spk_model(device=device, ckpt_path=self.spk_ckpt_path)
        if need_wer:
            if self.asr_zh is None:
                logger.info("[DiTAR_ORW] building zh ASR backend (paraformer-zh)")
                self.asr_zh = build_asr_model(lang="zh", ckpt_dir=None, device=device)
            if self.asr_en is None:
                logger.info("[DiTAR_ORW] building en ASR backend (faster-whisper)")
                self.asr_en = build_asr_model(lang="en", ckpt_dir=None, device=device)

    # ------------------------------------------------------------------
    # Reward helpers, split so "sim_and_wer" can reuse both.
    # ------------------------------------------------------------------
    def _reward_sim(self, gen_audios, raw_wav, raw_wav_mask):
        """Speaker similarity: cosine similarity between each rollout and the prompt (raw_wav) speaker."""
        # Force eval: spk_model is a submodule, so trainer's self.model.train() would flip it to train
        # mode; ECAPA-TDNN's BatchNorm1d then errors on the batch=1 prompt. eval uses running stats,
        # which also matches the official eval.
        self.spk_model.eval()
        # raw_wav: (B=1, T, 1); use the non-pad region as the reference speaker audio.
        prompt_wav_1d = raw_wav[0].squeeze(-1)               # (T,)
        prompt_audios = [prompt_wav_1d] * self.n_rollout      # reuse the same tensor -> embed once
        return run_wavlm_sim_local_batch(
            audios=gen_audios,
            prompt_audios=prompt_audios,
            gen_source_sr=self.spk_gen_sr, #16000
            prompt_source_sr=self.spk_prompt_sr, #16000
            device=gen_audios[0].device,
            model=self.spk_model,
        )

    def _reward_wer(self, gen_audios, gen_texts, cur_lang):
        """WER reward via local ASR (faster-whisper en / paraformer-zh),
        same backend and normalization as the official eval. Falls back to en when cur_lang is not zh/en."""
        asr_model = self.asr_zh if cur_lang == "zh" else self.asr_en
        return run_asr_local_batch(
            audios=gen_audios,
            ref_texts=gen_texts,
            lang=cur_lang,
            model=asr_model,
        )

    @staticmethod
    def _grpo_normalize(scores, device, B):
        """Within-group GRPO normalization: reshape scores to (B, n_rollout) then (x - mean)/(std + eps).
        Returns (normalized weights (B, n_rollout), group std (B,1)). std=0 means this reward gives no advantage signal."""
        w = torch.tensor(scores, dtype=torch.float32, device=device).view(B, -1)
        mean = w.mean(dim=1, keepdim=True)
        std = w.std(dim=1, keepdim=True)
        return (w - mean) / (std + 1e-8), std

    # ------------------------------------------------------------------
    # Main forward — drop-in replacement for DiTAR.forward(...). Same kwargs;
    # ORW mostly needs ``text`` and a prompt waveform.
    # ------------------------------------------------------------------
    def forward(
        self,
        raw_wav,                # (B, T_audio, 1) — used as the *prompt* for rollout
        raw_wav_mask,           # (B, T_audio) bool
        text,                   # list[str] of length B
        text_lengths,           # (B,) int tensor
        text_padding_mask=None,
        gen_text = None,
        lang = None,            # list[str] of length B ("zh"/"en"), inferred by the dataset from the audio path
        **_,
    ):
        device = raw_wav.device
        B = len(text)  # only B=1 is implemented

        cur_lang = lang[0] if isinstance(lang, (list, tuple)) else lang

        # ---- 1) rollout: n_rollout candidates in a single batched euler forward ----
        self.infer.load_state_dict(self.policy.state_dict())
        self.infer.eval()
        loss_cfg_strength = self.cfg_strength if self.loss_cfg else None

        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        gen_latents, _ = self.infer.sample_batch(
            prompt_wav=raw_wav,
            prompt_wav_mask=raw_wav_mask,
            text=text,
            text_lengths=text_lengths,
            text_padding_mask=None,
            n_rollout=self.n_rollout,
            max_seq_length=self.max_seq_length,
            min_seq_length=self.min_seq_length,
            steps=self.sample_steps,
            cfg_strength=self.cfg_strength,
            sway_sampling_coef=self.sway_sampling_coef,
            seed=seed,
            sample_strategy="stage",
        )
        # vocoder frozen -> decode serially; gen_audios[i]: (1, 1, T_wav_i).
        gen_audios = [self.policy.generator.decode(g) for g in gen_latents]
        gen_texts = [gen_text] * self.n_rollout

        # ---- 2) reward ("wer" / "sim" / "sim_and_wer") ----
        sim_rewards = None
        wer_rewards = None
        if self.reward_type in ("sim", "sim_and_wer"):
            sim_rewards = self._reward_sim(gen_audios, raw_wav, raw_wav_mask)
        if self.reward_type in ("wer", "sim_and_wer"):
            wer_rewards = self._reward_wer(gen_audios, gen_texts, cur_lang)

        # ---- 3) aggregate weights ----
        # ``score`` == (1 - wer) for ASR, == speaker-sim for spk (both ~[0,1]).
        score_device = gen_audios[0].device
        if self.reward_type == "sim_and_wer":
            # Normalize each reward within its group (z-score) *before* adding, not after: the two
            # rewards have different scales/variances, so combining first would let the high-variance
            # WER dominate and make SIM irrelevant.
            sim_scores = [r["score"] for r in sim_rewards]
            wer_scores = [r["score"] for r in wer_rewards]
            sim_w, sim_std = self._grpo_normalize(sim_scores, score_device, B)
            wer_w, wer_std = self._grpo_normalize(wer_scores, score_device, B)
            if self.rm_method == "exp_softmax":       # exp(τr): group softmax
                sim_raw = torch.tensor(sim_scores, dtype=torch.float32, device=score_device).view(B, -1)
                wer_raw = torch.tensor(wer_scores, dtype=torch.float32, device=score_device).view(B, -1)
                sim_s = torch.softmax(self.alpha * sim_raw, dim=1) * self.n_rollout
                wer_s = torch.softmax(self.alpha * wer_raw, dim=1) * self.n_rollout
                weights = (self.wer_weight * wer_s + self.sim_weight * sim_s) / (
                    self.wer_weight + self.sim_weight + 1e-8
                )
            elif self.rm_method == "exp":             # exp(τA): centered-advantage exponential
                combined = self.wer_weight * wer_w + self.sim_weight * sim_w
                weights = torch.exp(self.alpha * combined)
            else:                                     # grpo (default): signed z-score sum A/σ
                weights = self.wer_weight * wer_w + self.sim_weight * sim_w
            # acc = per-rollout mean of the two raw scores; r_std = mean of the two group stds (batch-skip gate only).
            # A reward with std=0 contributes no advantage; skip the batch only when both degrade (r_std~=0).
            acc = torch.tensor([(s + w) / 2.0 for s, w in zip(sim_scores, wer_scores)],dtype=torch.float32, device=score_device,)
            r_std = (wer_std + sim_std) / 2.0
        else:
            rewards = sim_rewards if self.reward_type == "sim" else wer_rewards
            acc = torch.tensor([r["score"] for r in rewards], dtype=torch.float32).to(score_device)
            weights = torch.tensor([r["score"] for r in rewards], dtype=torch.float32).to(score_device)
            weights = weights.view(B, self.n_rollout)
            r_mean = weights.mean(dim=1, keepdim=True)
            r_std = weights.std(dim=1, keepdim=True)
            if self.rm_method == "grpo":              # A/σ
                weights = (weights - r_mean) / (r_std + 1e-8)
            elif self.rm_method == "exp":             # exp(τr)
                weights = torch.exp(self.alpha * weights)
            elif self.rm_method == "exp_softmax":     # exp(τr): group softmax
                weights = torch.softmax(self.alpha * weights, dim=1) * self.n_rollout

        if self.rm_method == "grpo":
            r_std_scalar = r_std.mean().to(device)
            if r_std_scalar.item() == 0:
                return {"r_std": r_std_scalar,}

        # ---- 4) prepare data: prepend the prompt (raw_wav) to each generated audio, then pad ----
        raw_wav_sq = raw_wav.squeeze(-1)
        new_wavs = [torch.cat([raw_wav_sq, a.squeeze(1)], dim=1) for a in gen_audios]
        lengths = [w.shape[1] for w in new_wavs]
        max_new_wav_length = max(lengths)

        new_wav_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        new_wav_mask = ~make_pad_mask(new_wav_lengths, max_len=max_new_wav_length)

        padded_new_wavs = []
        for one_wav in new_wavs:
            pad_len = max_new_wav_length - one_wav.shape[1]
            padded_new_wavs.append(F.pad(one_wav, (0, pad_len), value=0.0))
        new_wavs = torch.stack(padded_new_wavs, dim=0)

        # replicate text / text_lengths to n_rollout copies, aligned with new_wavs
        rep_text = [t for t in text for _ in range(self.n_rollout)]
        rep_text_lengths = text_lengths.repeat_interleave(self.n_rollout)

        # ---- 5) policy forward ----
        policy_out = self.policy.forward_orw(
            raw_wav=new_wavs.transpose(1, 2),
            raw_wav_mask=new_wav_mask,
            text=rep_text,
            text_lengths=rep_text_lengths,
            loss_cfg_strength=loss_cfg_strength,   # None -> pure conditional (default)
        )
        # ---- 6) reference forward (no grad, shared time/x0) ----
        with torch.no_grad():
            ref_out = self.ref.forward_orw(
                raw_wav=new_wavs.transpose(1, 2),
                raw_wav_mask=new_wav_mask,
                text=rep_text,
                text_lengths=rep_text_lengths,
                time=policy_out["time"],
                x0=policy_out["x0"],
                vae_features=policy_out["vae_features"],
                vae_padding_mask=policy_out["vae_padding_mask"],
                loss_cfg_strength=loss_cfg_strength,  # same scale as policy + shared time -> matching alpha schedule
            )

        pred = policy_out["pred"]                 # (B', T, D)
        target = policy_out["target"]             # (B', T, D)
        ref_pred = ref_out["pred"]                # (B', T, D)
        mask = policy_out["vae_padding_mask"]     # (B', T) bool

        # ---- 6.5) mask out the prompt segment; compute loss only on gen_audios' vae frames ----
        # Same convention as extract_vae_features: valid VAE frames = ceil(L / hop_length).
        raw_wav_len = raw_wav.shape[1]
        raw_vae_len = math.ceil(raw_wav_len / self.policy.generator.hop_length)
        gen_frame_mask = ( torch.arange(mask.shape[1], device=mask.device) >= raw_vae_len)  # (T_padded,)  bool
        mask = mask & gen_frame_mask

        # ---- 7) loss ----
        cfm_loss, kl_value, pred_loss_unweighted = self._compute_orw_loss(
            pred=pred, ref_pred=ref_pred, target=target, mask=mask, weights=weights,
        )

        return {
            "diff_loss": cfm_loss,
            "kl_loss": kl_value.detach(),
            "pred_loss": pred_loss_unweighted.detach(),
            "implicit_acc": acc.mean(),
            "r_std": r_std.mean().to(device), # (B,1) -> 0-dim scalar, same format as loss
            "reward_per_rollout": acc.detach().reshape(B, self.n_rollout),
            "weights_per_rollout": weights.detach().reshape(B, self.n_rollout),

            "stop_loss": torch.tensor(0.0, device=device),
            "stop_acc": torch.tensor(0.0, device=device),
            "stop_sentence_acc": torch.tensor(0.0, device=device),
            "ar_l1_loss": torch.tensor(0.0, device=device),
            "vae_projected_l1_loss": torch.tensor(0.0, device=device),
        }

    # ------------------------------------------------------------------
    # Loss kernel: weighted MSE + β·KL (policy vs ref velocity), over valid
    # frames only. L2 (MSE) to match base-SFT geometry (the source DiTAR uses L1).
    # ------------------------------------------------------------------
    def _compute_orw_loss(self, pred, ref_pred, target, mask, weights):
        def _masked_mse(a, b):
            return ((a.float() - b.float()) ** 2).mean(dim=-1).masked_select(mask).mean()
        pred_mse = ((pred.float() - target.float()) ** 2).mean(dim=-1)    # (B', T)
        pred_loss_unweighted = pred_mse.masked_select(mask).mean()
        weighted = pred_mse * weights.unsqueeze(-1)                       # broadcast weight per sample
        weighted_loss = weighted.masked_select(mask).mean()

        kl_loss = _masked_mse(pred, ref_pred)
        cfm_loss = self.beta * kl_loss + weighted_loss
        return cfm_loss, kl_loss, pred_loss_unweighted


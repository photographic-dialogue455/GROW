"""DiTAR_SDE: Flow-GRPO (SDE) reward fine-tuning — the DiTAR-GRPO baseline.

Subclasses DiTAR_ORW (reward / checkpoint / policy-ref-infer scaffolding),
overriding only ``__init__`` (SDE hyperparams) and ``forward`` (SDE-GRPO step):
rollout n_rollout candidates with a stochastic SDE step over a short random window
of early denoising steps, group-normalize advantage, then REINFORCE loss
mean(-advantage · logπ) + β · KL over the recorded transitions.

``noise_level=0`` makes the SDE step reduce exactly to Euler, so outside the SDE
window the rollout is step-for-step identical to deterministic ``sample_batch``.
"""

from __future__ import annotations

import math
import random
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from DiTAR.model.ditar.dit_ar_orw import DiTAR_ORW

logger = logging.getLogger(__name__)


class DiTAR_SDE(DiTAR_ORW):
    """Flow-GRPO (SDE) fine-tuning wrapper. Inherits DiTAR_ORW's sub-models,
    reward backends, and checkpoint IO; overrides only ``__init__`` + ``forward``
    plus a few SDE helpers (all taking ``model`` as an arg)."""

    def __init__(self, ditar_cfg, tokenizer):
        super().__init__(ditar_cfg=ditar_cfg, tokenizer=tokenizer)

        # ---- SDE / Flow-GRPO hyperparams (reuse the orw config block) ----
        sde_cfg = self.orw_cfg
        self.noise_level = float(sde_cfg.get("noise_level", 0.8))   # sigma_t = sigma_prev * sin(noise_level * pi/2)
        self.sde_size = int(sde_cfg.get("sde_size", 3))            # number of consecutive stochastic SDE steps in the first half
        self.sde_sigma = float(sde_cfg.get("sde_sigma", 0.0))      # 0 reduces to exact Euler when noise_level=0
        self.sample_strategy = str(sde_cfg.get("sample_strategy", "stage"))  # rollout CFG schedule
        self.eos_threshold = float(sde_cfg.get("eos_threshold", 0.5))        # stop when softmax[:,1] > eos_threshold

    def _compute_sigma_t(self, t):
        """flow-matching sigma_t."""
        return 1 - (1 - self.sde_sigma) * t

    def sde_onestep_with_logprob(
        self,
        model_output: torch.FloatTensor,   # vt (velocity field)
        time,                              # current time T_t
        prev_time,                         # next time T_{t+1}
        sample: torch.FloatTensor,         # current sample x_t
        noise_level: float = 0.8,
        prev_sample: Optional[torch.FloatTensor] = None,  # x_{t+1}; passed in during training, None during sampling
    ):
        """One reverse SDE step. Returns (x_{t+1}, log_prob, prev_sample_mean, std_dev_t).

        Convention: x_0 is noise, x_1 is data. bf16 overflows at prev_sample_mean, so cast to fp32.
        """
        model_output = model_output.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()

        sigma = self._compute_sigma_t(time)
        sigma_prev = self._compute_sigma_t(prev_time)

        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)   # sigma_t in the paper
        noise_estimate = sample - (1 - sigma) * model_output           # predicted x_0 (noise)
        pred_original_sample = sample + model_output * sigma           # predicted x_1 (data)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
            sigma_prev ** 2 - std_dev_t ** 2
        )

        if prev_sample is None:
            variance_noise = torch.randn(
                model_output.shape, device=model_output.device, dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # Drop all constant terms (std_dev_t is constant within the window, so it doesn't affect the gradient).
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        # Mean over all dims except the batch dim.
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    # ---- LocDiT velocity (CFG): shared by rollout and loss, takes ``model`` as an arg ----
    def _locdit_cond_uncond(self, model, x, time_vec, ctx_raw, historical_patch, need_uncond):
        """Run LocDiT once; returns (pred_cond, pred_uncond | None).

        Matches the archived ``sample_batch`` fn: LocDiT input = [ctx | history | noisy],
        all-ones mask; uncond zeros out only the (projected) ctx.

        Shapes (rollout: B=active samples TB; loss: B=number of transitions N):
          x: (B, patch_size, C); time_vec: (B,);
          ctx_raw: (B, 1, D_model) before ctx_to_locdit; historical_patch: (B, window, C).
        """
        B = x.shape[0]
        device = x.device
        window = model.LocDiT_cfg.history_vae_window_size

        hist_emb = model.hist_to_locdit(historical_patch)
        ctx = model.ctx_to_locdit(ctx_raw)
        x_emb = model.noisy_to_locdit(x)
        time_embed = model.time_embedding(time_vec)

        locdit_mask = torch.ones(
            (B, 1 + window + model.patch_size), dtype=torch.bool, device=device,
        )

        cond_input = torch.cat([ctx, hist_emb, x_emb], dim=1)
        pred = model.LocDiT(x=cond_input, t=time_embed, mask=locdit_mask, patch_size=model.patch_size)
        if not need_uncond:
            return pred, None

        uncond_input = torch.cat([torch.zeros_like(ctx), hist_emb, x_emb], dim=1)
        null_pred = model.LocDiT(x=uncond_input, t=time_embed, mask=locdit_mask, patch_size=model.patch_size)
        return pred, null_pred

    @staticmethod
    def _stage_alpha(t, cur_alpha_outer, sample_strategy):
        """CFG alpha schedule over ODE time t (replicates sample_batch).

        stage: t<0.3 ramps up (1->alpha), t>0.7 = 1.0, middle = 0.2; other strategies stay at outer.
        Returns 0 when cur_alpha_outer<1e-5. Always returns a Python float (t may be a 0-d tensor;
        take the scalar explicitly to avoid a downstream torch.tensor(<tensor>) UserWarning).
        """
        t = float(t)
        if cur_alpha_outer < 1e-5:
            return 0.0
        if sample_strategy == "stage":
            stage_pre, stage_suf = 0.3, 0.7
            if t <= stage_pre:
                return (cur_alpha_outer - 1.0) * t / stage_pre + 1.0
            elif t >= stage_suf:
                return 1.0
            else:
                return 0.2
        return float(cur_alpha_outer)

    # ---- loss stage: recompute velocity for a batch of SDE transitions and get log_prob ----
    def _solver(self, model, inputs, time_vec, condition, pre_context, alpha):
        """CFG velocity: pred_cond + (pred_cond - pred_uncond) * alpha.

        Always computes uncond, then linearly mixes by the recorded per-step alpha (alpha=0 reduces to pred_cond).
        """
        pred, null_pred = self._locdit_cond_uncond(
            model, inputs, time_vec, ctx_raw=condition, historical_patch=pre_context, need_uncond=True,
        )
        alpha_ = alpha.view(-1, 1, 1)
        return pred + (pred - null_pred) * alpha_

    def compute_logprob(self, model, inputs, outputs, timesteps, prev_times, condition, pre_context, alpha):
        """Compute (log_prob, prev_sample_mean) for a batch of transitions.

        inputs/outputs: (N, patch_size, C); timesteps/prev_times/alpha: (N,);
        condition: (N, 1, D_model); pre_context: (N, window, C).
        """
        velocity = self._solver(model, inputs, timesteps, condition, pre_context, alpha)
        _, log_prob, prev_sample_mean, _ = self.sde_onestep_with_logprob(
            model_output=velocity,
            time=timesteps.view(-1, 1, 1),
            prev_time=prev_times.view(-1, 1, 1),
            sample=inputs,
            noise_level=self.noise_level,
            prev_sample=outputs,
        )
        return log_prob, prev_sample_mean

    # ---- Rollout: replicates sample_batch, replacing each patch's inner Euler with an explicit SDE step,
    # and recording the transitions of a random sde_size-step window in the first half ----
    @torch.no_grad()
    def sample_batch_with_trajectory(
        self,
        model,
        prompt_wav,                # (1, T_audio, 1)
        prompt_wav_mask,           # (1, T_audio) bool
        text,                      # list[str] length 1
        text_lengths,              # (1,) int
        n_rollout: int = 4,
        max_seq_length: int = 155,
        min_seq_length: int = 10,
        steps: int = 32,
        cfg_strength: float = 1.5,
        sway_sampling_coef=None,
        seed: int = 666,
    ):
        """Same-prompt, batched ``n_rollout`` sampling + SDE trajectory recording.

        Returns:
          gen_latents_list : list[(1, T_i, C)]  VAE latents of the n_rollout candidates;
          prompt_vae       : (1, T_prompt, C);
          trajectories     : list[dict]  SDE transitions recorded per candidate (fields are lists,
            elements are single-sample tensors: input/output (patch_size, C), condition (1, D_model),
            pre_context (window, C), timesteps/prev_times/alpha scalars).
        """
        torch.manual_seed(seed)
        model.eval()
        device = prompt_wav.device

        # ---- 1) encode the prompt once (same as sample_batch) ----
        if model.mel_spec_type == "semanticvae":
            from DiTAR.model.autoencoder_kl import extract_vae_features
            prompt_wav = prompt_wav.squeeze(-1).unsqueeze(1)
            if prompt_wav.dim() == 4:
                prompt_wav = prompt_wav.squeeze(1)
            prompt_vae, prompt_vae_mask, _ = extract_vae_features(model.generator, prompt_wav, prompt_wav_mask)
        else:
            raise NotImplementedError(f"sample_batch_with_trajectory does not support mel_spec_type={model.mel_spec_type}")

        # ---- 2) replicate prompt + text to n_rollout copies along the batch dim ----
        B = n_rollout
        prompt_vae = prompt_vae.expand(B, -1, -1).contiguous()
        prompt_vae_mask = prompt_vae_mask.expand(B, -1).contiguous()
        rep_text = [text[0] for _ in range(B)]
        rep_text_lengths = text_lengths.repeat(B)
        text_max_length = rep_text_lengths.max()

        # ---- 3) initial AR input (same as sample_batch) ----
        (
            ar_inputs_embed,
            ar_padding_mask,
            _text_emb,
            bpe_padding_mask,
            _vae_features,
            _vae_padding_mask,
            _vae_projected,
            _vae_aggregation,
            _vae_aggregation_mask,
            modality_type_ids,
        ) = model.get_ar_input(
            text=rep_text, text_max_length=text_max_length,
            vae_features=prompt_vae, vae_mask=prompt_vae_mask,
        )

        # Inject noise into a random run of sde_size consecutive ODE steps in the first half.
        # t_grid has steps+1 points => steps Euler steps.
        sde_start = random.randint(0, max(0, steps // 2 - self.sde_size))

        # ---- 4) AR loop + per-patch SDE sampling + trajectory recording ----
        window = model.LocDiT_cfg.history_vae_window_size
        vae_results_per_sample: list = [None for _ in range(B)]
        trajectories = [
            {"input": [], "output": [], "timesteps": [], "prev_times": [],
             "condition": [], "pre_context": [], "alpha": []}
            for _ in range(B)
        ]
        active_idx = torch.arange(B, device=device)

        cur_inputs_embed = ar_inputs_embed
        cur_padding_mask = ar_padding_mask
        cur_modality_type_ids = modality_type_ids
        vae_results = torch.empty((B, 0, model.audio_channels), dtype=ar_inputs_embed.dtype, device=device)

        for step in range(max_seq_length):
            TB = cur_inputs_embed.shape[0]
            if TB == 0:
                break

            h_predict, _ = model.causalAR.inference(
                inputs_embed=cur_inputs_embed,
                padding_mask=cur_padding_mask,
                modality_type_ids=cur_modality_type_ids,
            )
            h_last = h_predict[:, -1, :].unsqueeze(1)             # (TB, 1, D_model)

            stop_logit = model.eos_head(h_last.squeeze(1))         # (TB, 2)
            stop_prob = F.softmax(stop_logit, dim=-1)[:, 1]        # (TB,)

            # frame-level alpha cosine decay (same as sample_batch).
            cur_alpha_outer = cfg_strength
            if self.sample_strategy == "stage" and cfg_strength > 1.5:
                cfg_hz = 20
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5
            elif self.sample_strategy == "apg" and cfg_strength > 1.5:
                cfg_hz = 10
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5

            # independent noise start per active sample; t_grid same as sample_batch (with sway).
            y0 = torch.randn(TB, model.patch_size, model.audio_channels, device=device, dtype=h_last.dtype)
            t_grid = torch.linspace(0, 1, steps + 1, device=device, dtype=y0.dtype)
            if sway_sampling_coef is not None:
                t_grid = t_grid + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_grid) - 1 + t_grid)

            # history VAE window (same as sample_batch).
            if vae_results.numel() == 0:
                historical_patch = prompt_vae[active_idx, -window:, :]
            else:
                current_length = vae_results.shape[1]
                if current_length < window:
                    historical_patch = torch.cat(
                        [prompt_vae[active_idx, -(window - current_length):, :], vae_results], dim=1,
                    )
                else:
                    historical_patch = vae_results[:, -window:, :]

            # ---- explicit SDE/Euler integration of this patch, recording in-window transitions ----
            sample = y0 
            n_euler = t_grid.shape[0] - 1
            win_input, win_output, win_t, win_tn, win_alpha = [], [], [], [], []
            for i in range(n_euler):
                t = t_grid[i]
                t_next = t_grid[i + 1]
                time_vec = t.unsqueeze(0).expand(TB)               # (TB,)

                need_uncond = cur_alpha_outer >= 1e-5
                pred_cond, pred_uncond = self._locdit_cond_uncond(
                    model, sample, time_vec, ctx_raw=h_last, historical_patch=historical_patch,
                    need_uncond=need_uncond,
                )
                alpha_i = self._stage_alpha(t, cur_alpha_outer, self.sample_strategy)
                if need_uncond and alpha_i != 0.0:
                    velocity = pred_cond + (pred_cond - pred_uncond) * alpha_i
                else:
                    velocity = pred_cond

                in_window = (sde_start <= i < sde_start + self.sde_size)
                cur_noise_level = self.noise_level if in_window else 0.0
                if in_window:
                    win_input.append(sample)                       # x_t (before step)

                sample, _, _, _ = self.sde_onestep_with_logprob(
                    model_output=velocity, time=t, prev_time=t_next,
                    sample=sample, noise_level=cur_noise_level, prev_sample=None,
                )

                if in_window:
                    win_output.append(sample)                      # x_{t+1} (after step)
                    win_t.append(t)
                    win_tn.append(t_next)
                    win_alpha.append(alpha_i)

            sampled = sample                                       # final x_1 = (TB, patch_size, C)
            vae_results = torch.cat((vae_results, sampled), dim=1)

            # ---- distribute this patch's recorded window transitions to each active candidate ----
            # Include samples that stop this step too: they did generate this patch.
            for j in range(TB):
                orig = active_idx[j].item()
                tr = trajectories[orig]
                for k in range(len(win_input)):
                    tr["input"].append(win_input[k][j].float().clone())
                    tr["output"].append(win_output[k][j].float().clone())
                    tr["condition"].append(h_last[j].clone())          # (1, D_model)
                    tr["pre_context"].append(historical_patch[j].clone())  # (window, C)
                    tr["timesteps"].append(win_t[k].detach().clone())
                    tr["prev_times"].append(win_tn[k].detach().clone())
                    tr["alpha"].append(
                        torch.tensor(win_alpha[k], device=device, dtype=sampled.dtype)
                    )

            # ---- per-sample stop check (same as sample_batch) ----
            if step >= min_seq_length:
                stop_now = stop_prob > self.eos_threshold
            else:
                stop_now = torch.zeros(TB, dtype=torch.bool, device=device)

            if stop_now.any():
                stopped_local = stop_now.nonzero(as_tuple=False).flatten().tolist()
                for i_local in stopped_local:
                    orig_i = active_idx[i_local].item()
                    vae_results_per_sample[orig_i] = vae_results[i_local].clone()

                keep = ~stop_now
                cur_inputs_embed = cur_inputs_embed[keep]
                cur_padding_mask = cur_padding_mask[keep]
                cur_modality_type_ids = cur_modality_type_ids[keep]
                vae_results = vae_results[keep]
                active_idx = active_idx[keep]
                sampled = sampled[keep]

                if cur_inputs_embed.shape[0] == 0:
                    break

            # ---- encode the freshly generated patch back into the AR input (same as sample_batch) ----
            input_vae_patch_emb = model.aggregation_inp_linear(sampled)
            vae_patch_mask = torch.ones((sampled.shape[0], model.patch_size), dtype=torch.bool, device=device)
            if model.patch_size == 1:
                aggregation_emb = input_vae_patch_emb
            else:
                aggregation_emb, _ = model.aggregation_encoder(input_vae_patch_emb, padding_mask=vae_patch_mask)

            cur_inputs_embed = torch.cat([cur_inputs_embed, aggregation_emb], dim=1)
            cur_padding_mask = torch.cat(
                [cur_padding_mask, torch.ones((cur_inputs_embed.shape[0], 1), device=device, dtype=torch.bool)], dim=1,
            )
            cur_modality_type_ids = torch.cat(
                [cur_modality_type_ids, torch.ones((cur_inputs_embed.shape[0], 1), dtype=torch.int64, device=device)], dim=1,
            )

        # Snapshot samples that never stopped within max_seq_length.
        for i_local, orig_i in enumerate(active_idx.tolist()):
            if vae_results_per_sample[orig_i] is None:
                vae_results_per_sample[orig_i] = vae_results[i_local].clone()

        gen_latents_list = [v.unsqueeze(0) for v in vae_results_per_sample]
        return gen_latents_list, prompt_vae, trajectories

    # ---- Main forward: drop-in replacement for DiTAR.forward, same trainer interface as ORW ----
    def forward(
        self,
        raw_wav,                # (B, T_audio, 1) — used as the rollout prompt
        raw_wav_mask,           # (B, T_audio) bool
        text,                   # list[str] length B (designed for B=1 single prompt, same as ORW)
        text_lengths,           # (B,) int
        text_padding_mask=None,
        gen_text=None,
        lang=None,
        **_,
    ):
        device = raw_wav.device
        cur_lang = lang[0] if isinstance(lang, (list, tuple)) else lang

        # ---- 1) rollout (+ SDE trajectory) ----
        self.infer.load_state_dict(self.policy.state_dict())
        self.infer.eval()

        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        gen_latents, _prompt_vae, trajectories = self.sample_batch_with_trajectory(
            self.infer,
            prompt_wav=raw_wav,
            prompt_wav_mask=raw_wav_mask,
            text=text,
            text_lengths=text_lengths,
            n_rollout=self.n_rollout,
            max_seq_length=self.max_seq_length,
            min_seq_length=self.min_seq_length,
            steps=self.sample_steps,
            cfg_strength=self.cfg_strength,
            sway_sampling_coef=self.sway_sampling_coef,
            seed=seed,
        )

        # vocoder decode (same as ORW).
        gen_audios = [self.policy.generator.decode(g) for g in gen_latents]
        gen_texts = [gen_text] * self.n_rollout

        # ---- 2) reward (reuse ORW reward infra) ----
        sim_rewards = None
        wer_rewards = None
        if self.reward_type in ("sim", "sim_and_wer"):
            sim_rewards = self._reward_sim(gen_audios, raw_wav, raw_wav_mask)
        if self.reward_type in ("wer", "sim_and_wer"):
            wer_rewards = self._reward_wer(gen_audios, gen_texts, cur_lang)

        # ---- 3) advantage aggregation (same as ORW; B=1 single prompt) ----
        B = 1
        score_device = gen_audios[0].device
        if self.reward_type == "sim_and_wer":
            sim_scores = [r["score"] for r in sim_rewards]
            wer_scores = [r["score"] for r in wer_rewards]
            sim_w, sim_std = self._grpo_normalize(sim_scores, score_device, B)
            wer_w, wer_std = self._grpo_normalize(wer_scores, score_device, B)
            weights = self.wer_weight * wer_w + self.sim_weight * sim_w
            acc = torch.tensor(
                [(s + w) / 2.0 for s, w in zip(sim_scores, wer_scores)],
                dtype=torch.float32, device=score_device,
            )
            r_std = (wer_std + sim_std) / 2.0
        else:
            rewards = sim_rewards if self.reward_type == "sim" else wer_rewards
            acc = torch.tensor([r["score"] for r in rewards], dtype=torch.float32).to(score_device)
            weights = torch.tensor([r["score"] for r in rewards], dtype=torch.float32).to(score_device)
            weights = weights.view(B, self.n_rollout)
            r_mean = weights.mean(dim=1, keepdim=True)
            r_std = weights.std(dim=1, keepdim=True)
            # DiTAR-GRPO baseline: signed group-normalized advantage A/σ。
            weights = (weights - r_mean) / (r_std + 1e-8)

        r_std_scalar = r_std.mean().to(device)
        # All rewards in the group equal (std=0) => no advantage signal, let the trainer resample the batch.
        if r_std_scalar.item() == 0:
            return {"r_std": r_std_scalar}

        weights = weights.view(-1)  # (n_rollout,)

        # ---- 4) flatten all candidates' SDE transitions into one flat batch ----
        # (B=1 single prompt, no pad+mask needed: flat mean == masked mean)
        counts = [len(trajectories[r]["input"]) for r in range(self.n_rollout)]
        total = sum(counts)
        if total == 0:
            # Should not happen (sde_size>=1 => each candidate has at least sde_size transitions).
            # If it does (e.g. sde_size=0), return r_std=0 so the trainer marks this batch invalid and
            # resamples, matching the std==0 early-exit and avoiding an empty loss that would crash backward.
            return {"r_std": torch.zeros((), device=device)}

        all_inputs = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["input"]], dim=0
        )                                                          # (N, patch_size, C)
        all_outputs = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["output"]], dim=0
        )
        all_condition = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["condition"]], dim=0
        )                                                          # (N, 1, D_model)
        all_pre_context = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["pre_context"]], dim=0
        )                                                          # (N, window, C)
        all_timesteps = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["timesteps"]], dim=0
        ).view(-1).to(device)                                      # (N,)
        all_prev_times = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["prev_times"]], dim=0
        ).view(-1).to(device)
        all_alpha = torch.stack(
            [x for r in range(self.n_rollout) for x in trajectories[r]["alpha"]], dim=0
        ).view(-1).to(device)

        # Each transition's advantage = the advantage of its owning candidate (broadcast).
        weight_vec = torch.cat(
            [weights[r].reshape(1).expand(counts[r]) for r in range(self.n_rollout)], dim=0
        ).to(device)                                               # (N,)

        # ---- 5) policy / reference each recompute log_prob and prev_sample_mean ----
        # Intentional (not a bug): the SDE/Flow-GRPO policy gradient updates ONLY the LocDiT denoiser
        # (and its projection layers ctx/hist/noisy_to_locdit and time_embedding). The AR backbone,
        # aggregation_encoder, eos_head, and text_embedding get no gradient — the condition/pre_context
        # they produce is recorded and replayed as constants inside the @torch.no_grad rollout
        # (compute_logprob only runs LocDiT). On the trainer side, find_unused_parameters=True already
        # handles these no-grad params.
        log_prob, mean_policy = self.compute_logprob(
            self.policy, all_inputs, all_outputs, all_timesteps, all_prev_times,
            all_condition, all_pre_context, all_alpha,
        )
        with torch.no_grad():
            _, mean_ref = self.compute_logprob(
                self.ref, all_inputs, all_outputs, all_timesteps, all_prev_times,
                all_condition, all_pre_context, all_alpha,
            )

        # ---- 6) loss = REINFORCE pg_loss + β · KL ----
        kl_loss = ((mean_policy - mean_ref) ** 2).mean(dim=tuple(range(1, mean_policy.ndim))).mean()
        pg_loss = (-log_prob * weight_vec).mean()
        loss = pg_loss + self.beta * kl_loss

        # Monitoring: pred_loss = mean -log_prob magnitude.
        pred_loss = (-log_prob).mean().detach()

        return {
            "diff_loss": loss,
            "kl_loss": kl_loss.detach(),
            "pred_loss": pred_loss,
            "implicit_acc": acc.mean(),
            "r_std": r_std.mean().to(device),
            # placeholder entries (same as ORW, so trainer / loss_weight can be reused).
            "reward_per_rollout": acc.detach().reshape(B, self.n_rollout),
            "weights_per_rollout": weights.detach().reshape(B, self.n_rollout),

            
            "stop_loss": torch.tensor(0.0, device=device),
            "stop_acc": torch.tensor(0.0, device=device),
            "stop_sentence_acc": torch.tensor(0.0, device=device),
            "ar_l1_loss": torch.tensor(0.0, device=device),
            "vae_projected_l1_loss": torch.tensor(0.0, device=device),
        }

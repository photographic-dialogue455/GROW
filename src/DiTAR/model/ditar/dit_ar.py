from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm

from DiTAR.model.ditar.input_embedding import TimestepEmbedding
from DiTAR.model.ditar.aggregation_encoder_qwen import AggregationEncoder_Qwen
from DiTAR.model.utils import get_mask_from_lengths
from DiTAR.model.autoencoder_kl import build_generator, extract_vae_features
from DiTAR.model.ditar.layers import Linear

@dataclass
class LocDiTConfig:
    name: Literal["DiT"] = "DiT"

    # Model
    model: dict = field(default_factory=dict)
    history_vae_window_size: int = 4

    # Training
    random_time: bool = False
    time_schedule: bool = False
    drop_cond_prob: float = 0.1

    # Inference
    odeint_kwargs: dict = field(default_factory=lambda: {
        "method": "euler",
    })

    def __post_init__(self):
        if self.name != "DiT":
            raise ValueError

class DiTAR(nn.Module):
    def __init__(
        self,
        ditar_cfg,
        tokenizer=None,
    ):
        super().__init__()
        self.ditar_cfg = ditar_cfg
        self.audio_channels = ditar_cfg.audio_channels
        self.dim = ditar_cfg.dim
        self.patch_size = ditar_cfg.patch_size
        self.text_vocab_size = tokenizer.vocab_size
        
        self.tokenizer = tokenizer

        # CausalAR
        if ditar_cfg.backbone.name == "Qwen":
            from DiTAR.model.backbones.casual_ar_qwen import CausalAR_Qwen
            self.causalAR = CausalAR_Qwen(
                version=ditar_cfg.backbone.version,
                qwen_config_path=ditar_cfg.backbone.qwen_config_path,
                pretrained_LM_path=ditar_cfg.backbone.pretrained_LM_path,
                load_pretrained_weights=ditar_cfg.backbone.load_pretrained_weights,
            )
        else:
            raise NotImplementedError

        self.mlp_hidden_dim = ditar_cfg.mlp_hidden_dim
        # Legacy MLP kept for checkpoint compatibility; not used in the forward path.
        self.vae_projector = nn.Sequential(
            nn.Linear(self.audio_channels, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.dim)
        )

        # Separate projections into LocDiT for ctx / history patch / noisy patch.
        self.ctx_to_locdit = Linear(self.dim, self.dim)
        self.hist_to_locdit = Linear(self.audio_channels, self.dim)
        self.noisy_to_locdit = Linear(self.audio_channels, self.dim)

        self.time_embedding = TimestepEmbedding(dim = self.dim,)

        # VAE aggregator
        if ditar_cfg.aggregation_encoder.name =="qwen":
            self.aggregation_encoder = AggregationEncoder_Qwen(
                **ditar_cfg.aggregation_encoder,
            )
        else:
            raise NotImplementedError

        self.aggregation_inp_linear = Linear(ditar_cfg.audio_channels, ditar_cfg.aggregation_encoder.hidden_size)


        # Local decoder
        if ditar_cfg.loc_decoder.name == "DiT":
            from DiTAR.model.backbones.dit import DiT
            self.LocDiT_cfg = LocDiTConfig(**ditar_cfg.loc_decoder)
            self.LocDiT = DiT(**self.LocDiT_cfg.model)
        else:
            raise NotImplementedError

        self.eos_head = nn.Linear(self.dim, 2)

        self.audio_type = ditar_cfg.audio_type
        self.mel_spec_type = ditar_cfg.vocoder.mel_spec_type
        self.feature_extractor_ckpt = ditar_cfg.vocoder.feature_extractor_ckpt
        if self.mel_spec_type == "semanticvae":
            self.generator = build_generator(path=self.feature_extractor_ckpt)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_ar_input(
        self, 
        text, 
        text_max_length, 
        vae_features, 
        vae_mask, 
    ):
        patch_size = self.patch_size
        device = self.device

        assert isinstance(text, list)
        # QwenTokenizer (BPE) is the only supported tokenizer.
        text_indices = self.tokenizer(text, padding=True, return_tensors="pt").to(device)
        text_emb = self.causalAR.model.embed_tokens(text_indices.input_ids)
        bpe_padding_mask = text_indices['attention_mask'].bool()  # valid tokens first, then padding

        # Right-pad VAE features so the length is a multiple of patch_size.
        B, L, D = vae_features.shape
        pad_length = (-L) % patch_size

        padded_vae_features = F.pad(
            vae_features, (0, 0, 0, pad_length)
        )  # (B, T_vae_padded, feature_dim)
        padded_vae_mask = F.pad(
            vae_mask, (0, pad_length), value=False
        )  # (B, T_vae_padded)

        # Project VAE features to model embedding dimension.
        padded_vae_features_projected = self.aggregation_inp_linear(padded_vae_features)

        # Aggregate patches if needed
        if patch_size == 1:
            vae_aggregation = padded_vae_features_projected
            vae_aggregation_mask = padded_vae_mask
        else:
            vae_aggregation, vae_aggregation_mask = self.aggregation_encoder(
                padded_vae_features_projected,
                padding_mask = padded_vae_mask,
            )  # (B, T/patch_size, D)

        # Concatenate text and VAE embeddings: (B, T_text + T_agg, D)
        inputs_embed = torch.cat([text_emb, vae_aggregation], dim=1)
        modality_type_ids = torch.cat([
            torch.zeros(*text_emb.shape[:-1], dtype=torch.int64, device=device), 
            torch.ones(*vae_aggregation.shape[:-1], dtype=torch.int64, device=device)
        ], dim=1)
        inputs_padding_mask = torch.cat((bpe_padding_mask, vae_aggregation_mask), dim=1)

        return (
            inputs_embed,
            inputs_padding_mask,
            text_emb,
            bpe_padding_mask,
            padded_vae_features, 
            padded_vae_mask, 
            padded_vae_features_projected, 
            vae_aggregation,
            vae_aggregation_mask, 
            modality_type_ids, 
        )
    
    def get_decoder_input(
        self,
        *,
        vae_features,
        vae_padding_mask,
        h_predict,
        noisy_input,
        apply_drop_cond: bool = True,   # forward: True; forward_orw passes False so policy/ref stay comparable
    ):
        B, n_patches, D_model = h_predict.shape
        history_vae_window_size = self.LocDiT_cfg.history_vae_window_size
        device = self.device

        # 1. ctx feature — current AR hidden state
        ctx = self.ctx_to_locdit(h_predict)
        if apply_drop_cond:
            ctx_drop_mask = (torch.rand(B, device=device) >= self.LocDiT_cfg.drop_cond_prob).to(ctx.dtype)
            ctx = ctx * ctx_drop_mask.reshape(B, 1, 1)
        folded_ctx = rearrange(ctx, 'b n_patches d_model -> (b n_patches) 1 d_model')

        # 2. History patches: left-pad window_size zeros so the first window is all zeros.
        dit_vae_projected = self.hist_to_locdit(vae_features)

        _left_pad = torch.zeros(
            (B, history_vae_window_size, D_model),
            device = device,
            dtype = dit_vae_projected.dtype
        )
        vae_projected__left_padded = torch.cat([
            _left_pad,
            dit_vae_projected,
        ], dim=1)  # (B, T_vae + window_size, D_model)

        _left_pad_mask = torch.zeros(
            (B, history_vae_window_size),
            dtype = torch.bool,
            device = device,
        )
        vae_padding_mask__left_padded = torch.cat([
            _left_pad_mask,
            vae_padding_mask,
        ], dim=1)  # (B, T_vae + window_size)

        # Sliding window over time: extract each patch's history window.
        hist_emb = vae_projected__left_padded.unfold(
            dimension = 1,
            size = history_vae_window_size,
            step = self.patch_size,
        )[:, :n_patches, ...]  # (B, n_patches, D_model, window_size)
        hist_msk = vae_padding_mask__left_padded.unfold(
            dimension = 1,
            size = history_vae_window_size,
            step = self.patch_size,
        )[:, :n_patches, ...]  # (B, n_patches, window_size)

        # Merge batch & time for LocDiT.
        hist_emb = rearrange(hist_emb, 'b n_patches d_model n_history_vae -> (b n_patches) n_history_vae d_model')
        hist_msk = rearrange(hist_msk, 'b n_patches n_history_vae -> (b n_patches) n_history_vae')

        # 3. Current noisy patch x_t → embed to D_model
        noisy_emb = self.noisy_to_locdit(noisy_input)
        noisy_emb = rearrange(noisy_emb, 'b (n_patches patch_size) d_model -> (b n_patches) patch_size d_model', patch_size=self.patch_size)
        folded_vae_mask = rearrange(vae_padding_mask, 'b (n_patches patch_size) -> (b n_patches) patch_size', patch_size=self.patch_size)

        # 4. Concatenate [ctx | history | noisy]
        combined_input = torch.cat((folded_ctx, hist_emb, noisy_emb), dim=1)

        ctx_mask = torch.ones((B * n_patches, 1), dtype=torch.bool, device=device)
        combined_mask = torch.cat((ctx_mask, hist_msk, folded_vae_mask), dim=1)

        return combined_input, combined_mask

    def forward(
        self,
        raw_wav,
        raw_wav_mask,
        text: list[str],
        text_lengths,
        text_padding_mask,
        **kwargs,
    ):
        device = raw_wav.device
        if self.audio_type == "vae_online":
            if self.mel_spec_type == "semanticvae":
                raw_wav = raw_wav.squeeze(-1).unsqueeze(1)
                vae_features, vae_padding_mask, _ = extract_vae_features(self.generator, raw_wav, raw_wav_mask)

        # Determine max text length and batch size
        text_max_length = text_lengths.max()
        B = vae_features.shape[0]

        # Prepare inputs
        (
            ar_inputs_embed,
            ar_padding_mask,
            text_emb,               # (B, T_text, D_model)
            bpe_padding_mask,       # None or (B, T_text)
            vae_features,           # Right padded to multiple of patch size (T_vae)
            vae_padding_mask,       
            vae_projected,          # (B, T_vae, D_model)
            vae_aggregation,        # (B, n_patches, D_model)
            vae_aggregation_mask, 
            modality_type_ids,
        ) = self.get_ar_input(
            text = text,
            text_max_length = text_max_length,
            vae_features = vae_features,
            vae_mask = vae_padding_mask,
        )
        text_max_length = bpe_padding_mask.shape[1]

        ar_pred = self.causalAR(
            inputs_embed = ar_inputs_embed,
            padding_mask = ar_padding_mask,
            modality_type_ids = modality_type_ids, 
        ) #(B,T,D)

        # Discard text prefix (except its last token); keep only generated patch states.
        # The last VAE patch was already stripped from the end of ar_inputs_embed.
        ar_pred = ar_pred[:, text_max_length - 1: -1]

        stop_hidden = ar_pred.detach()
        stop_logits = self.eos_head(stop_hidden)  # (B, n_patches, 2)

        vae_patch_nums = vae_aggregation_mask.sum(dim=1).long() # (B,)
        assert vae_patch_nums.max() == ar_pred.shape[1]
        vae_patch_nums_max = ar_pred.shape[1]
        stop_targets =   (~get_mask_from_lengths(vae_patch_nums - 1, max_len=vae_patch_nums_max)).long()  # 0s then 1s
        stop_mask = ~get_mask_from_lengths(vae_patch_nums, max_len=vae_patch_nums_max)  # False then True

        stop_logits_t = stop_logits.transpose(1, 2)
        stop_loss = F.cross_entropy(stop_logits_t, stop_targets, reduction='none')
        stop_ratio = stop_targets * (vae_patch_nums - 1).unsqueeze(1) + 1
        stop_ratio = stop_ratio.masked_fill(stop_mask, 0)
        stop_ratio = stop_ratio / vae_patch_nums.unsqueeze(1).float()
        stop_loss = (stop_loss * stop_ratio).sum(dim=1)
        stop_loss = stop_loss.mean()

        # ---- token-level stop accuracy on valid positions (no grad, monitoring only) ----
        with torch.no_grad():
            stop_pred = stop_logits.argmax(dim=-1)           # (B, n_patches)
            valid_position = ~stop_mask
            correct = (stop_pred == stop_targets) & valid_position
            stop_acc = correct.sum().float() / valid_position.sum().float()

        # ---- sentence-level stop accuracy (no grad, monitoring only) ----
        with torch.no_grad():
            wrong_at_valid = (stop_pred != stop_targets) & valid_position  # (B, n_patches)
            sentence_correct = ~wrong_at_valid.any(dim=1)                  # (B,) True = whole sentence correct
            stop_sentence_acc = sentence_correct.float().mean()

        # Flow matching: x_1 -> add noise -> x_0
        time = torch.rand((B,), dtype=ar_pred.dtype, device=device)
        t = time[..., None, None]  # (B, 1, 1)

        x_1 = vae_features  # (B, T_padded, D_vae), T_padded == n_patches * patch_size
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t) * x_0 + t * x_1
        flow = x_1 - x_0

        locdit_input, locdit_mask = self.get_decoder_input(
            vae_features = vae_features,
            vae_padding_mask = vae_padding_mask,
            h_predict = ar_pred,
            noisy_input = x_t,
        )

        time_embed = self.time_embedding(time)  # (B, D_model)
        pred = self.LocDiT(
            x = locdit_input,
            t = time_embed,
            mask = locdit_mask,
            patch_size = self.patch_size,
        )
        
        # Diffusion loss
        diff_loss = F.mse_loss(pred, flow, reduction='none')
        diff_loss = diff_loss[vae_padding_mask].mean()

        return {
            "diff_loss": diff_loss, 
            "stop_loss": stop_loss, 
            "stop_acc": stop_acc,
            "stop_sentence_acc": stop_sentence_acc,
        }

    def forward_orw(
        self,
        raw_wav,
        raw_wav_mask,
        text,                    # list[str]
        text_lengths=None,       # tensor[int] (B,)
        time=None,               # optional shared sampling time (B,)
        x0=None,                 # optional shared noise (B, T_padded, D_vae)
        vae_features=None,       # ORW: optional pre-extracted VAE latents shared policy<->ref
        vae_padding_mask=None,   # ORW: padding mask paired with the shared vae_features
        loss_cfg_strength=None,  # ORW: if given (>=1e-5), also CFG-blend the loss velocity (matches rollout); None = pure conditional
    ):
        """Flow-matching forward from pre-extracted VAE latents (ORW).

        Unlike ``forward``: returns per-frame velocity ``pred``/``target`` (caller weights
        by reward); accepts shared ``time``/``x0``/``vae_features`` so policy and ref compute
        the velocity on the same target and (time, x0) for comparable KL/MSE; disables ctx
        CFG drop (``apply_drop_cond=False``); stop/acc are not computed.
        """
        device = raw_wav.device
        # Extract latents only when not passed (policy path). Ref call passes the policy's
        # vae_features / vae_padding_mask so both use the same target and (time, x0).
        if vae_features is None:
            if self.mel_spec_type == "semanticvae":
                raw_wav = raw_wav.squeeze(-1).unsqueeze(1)  # (B, 1, T)
                vae_features, vae_padding_mask, _ = extract_vae_features(
                    self.generator, raw_wav, raw_wav_mask
                )
            else:
                raise NotImplementedError(
                    f"forward_orw only supports semanticvae, got {self.mel_spec_type}"
                )

        text_max_length = text_lengths.max()
        B = vae_features.shape[0]

        (
            ar_inputs_embed,
            ar_padding_mask,
            text_emb,
            bpe_padding_mask,
            vae_features,            # right-padded to multiple of patch_size
            vae_padding_mask,
            vae_projected,
            vae_aggregation,
            vae_aggregation_mask,
            modality_type_ids,
        ) = self.get_ar_input(
            text=text,
            text_max_length=text_max_length,
            vae_features=vae_features,
            vae_mask=vae_padding_mask,
        )
        text_max_length = bpe_padding_mask.shape[1]

        ar_pred = self.causalAR(
            inputs_embed=ar_inputs_embed,
            padding_mask=ar_padding_mask,
            modality_type_ids=modality_type_ids,
        )
        ar_pred = ar_pred[:, text_max_length - 1: -1]

        if time is None:
            time = torch.rand((B,), dtype=ar_pred.dtype, device=device)
        if x0 is None:
            x0 = torch.randn_like(vae_features)

        t = time[..., None, None]  # (B, 1, 1)
        x_t = (1 - t) * x0 + t * vae_features
        flow = vae_features - x0

        locdit_input, locdit_mask = self.get_decoder_input(
            vae_features=vae_features,
            vae_padding_mask=vae_padding_mask,
            h_predict=ar_pred,
            noisy_input=x_t,
            apply_drop_cond=False,   # ORW: disable get_decoder_input's internal random drop; keeps policy/ref comparable
        )

        time_embed = self.time_embedding(time)
        pred = self.LocDiT(
            x=locdit_input,
            t=time_embed,
            mask=locdit_mask,
            patch_size=self.patch_size,
        )

        # ---- ORW loss_cfg: CFG-blend the loss velocity with the 3-stage alpha schedule ----
        do_loss_cfg = loss_cfg_strength is not None and float(loss_cfg_strength) >= 1e-5
        if do_loss_cfg:
            uncond_input = locdit_input.clone()
            uncond_input[:, :1, :] = 0.0                       # zero only the ctx token (idx 0)
            null_pred = self.LocDiT(
                x=uncond_input,
                t=time_embed,
                mask=locdit_mask,
                patch_size=self.patch_size,
            )
            a_out = float(loss_cfg_strength)
            tt = time                                          # (B,), sampling t driving the stage schedule
            alpha = torch.full_like(tt, 0.2)
            alpha = torch.where(tt <= 0.3, (a_out - 1.0) * tt / 0.3 + 1.0, alpha)
            alpha = torch.where(tt >= 0.7, torch.ones_like(tt), alpha)
            pred = pred + (pred - null_pred) * alpha[:, None, None]

        return {
            "pred": pred,                          # (B, T_padded, D_vae)
            "target": flow,                        # (B, T_padded, D_vae)
            "time": time,                          # (B,)
            "x0": x0,                              # (B, T_padded, D_vae)
            "vae_padding_mask": vae_padding_mask,  # (B, T_padded) bool
            "vae_features": vae_features,          # (B, T_padded, D_vae)
        }

    @torch.no_grad()
    def sample(
        self,
        prompt_wav,
        prompt_wav_mask,
        text,
        text_lengths,
        max_seq_length = 250, #25s
        min_seq_length = 10, #1s
        steps = 32,
        cfg_strength = 1.5,
        sway_sampling_coef = None,
        seed: int = 666,
        sample_strategy: Literal["base", "stage", "apg"] = "stage",
        kv_cache: bool = False,
        eos_threshold: float = 0.5,
    ):
        """
        sample_strategy:
            "base"  : classic CFG — pred + (pred - null_pred) * alpha
            "stage" : staged CFG — cosine alpha decay along frames (cfg_hz=20) plus a
                      3-stage schedule along ODE time t (t<0.3 ramp, t>0.7 = 1.0, else 0.2).
            "apg"   : interface kept; falls back to base here.

        kv_cache:
            True  -> full prefix on step 1, then 1 new patch/step (position_ids rebuilt from
                     accumulated modality_type_ids).
            False -> re-run the whole ar_inputs_embed every step (slower).
        """
        torch.manual_seed(seed)
        self.eval()
        device = prompt_wav.device
        if self.mel_spec_type == "semanticvae":
            prompt_wav = prompt_wav.squeeze(-1).unsqueeze(1)
            if prompt_wav.dim() == 4:
                prompt_wav = prompt_wav.squeeze(1)
            prompt_vae, prompt_vae_mask, _ = extract_vae_features(self.generator, prompt_wav, prompt_wav_mask)

        B = prompt_vae.shape[0]
        text_max_length = text_lengths.max()

        (
            ar_inputs_embed,
            ar_padding_mask,        # padding mask over text + vae_aggregation
            text_emb,               # (B, T_text, D_model)
            bpe_padding_mask,       # None or (B, T_text)
            vae_features,
            vae_padding_mask,
            vae_projected,          # (B, T_vae, D_model)
            vae_aggregation,        # (B, n_patches, D_model)
            vae_aggregation_mask,
            modality_type_ids,
        ) = self.get_ar_input(
            text = text,
            text_max_length = text_max_length,
            vae_features = prompt_vae,
            vae_mask = prompt_vae_mask,
        )
        text_max_length = bpe_padding_mask.shape[1]

        # Accumulated output: decoded raw (unprojected) vae patches.
        vae_results = torch.empty((B, 0, self.audio_channels), dtype=ar_inputs_embed.dtype, device=device)

        # KV-cache state
        past_key_values = None
        cur_inputs_embed = ar_inputs_embed                # full prefix on step 1; only 1 new patch after that in kv_cache mode
        cur_modality_type_ids = modality_type_ids         # kept full length (ModelForMultiModal restores position counts from it)
        cur_padding_mask = ar_padding_mask                # kept full length

        for step in tqdm(range(max_seq_length), desc="Generating"):
            if kv_cache:
                h_predict, past_key_values = self.causalAR.inference(
                    inputs_embed = cur_inputs_embed,
                    padding_mask = cur_padding_mask,
                    modality_type_ids = cur_modality_type_ids,
                    past_key_values = past_key_values,
                    use_cache = True,
                )
            else:
                h_predict, _ = self.causalAR.inference(
                    inputs_embed = cur_inputs_embed,
                    padding_mask = cur_padding_mask,
                    modality_type_ids = cur_modality_type_ids,
                )
            # Last-position hidden state = this patch's ctx h_i.
            h_last = h_predict[:, -1, :].unsqueeze(1)     # (B, 1, D_model)

            # EOS check: eos_head trained with 2-class CE in forward, so use softmax[:, 1].
            stop_logit = self.eos_head(h_last.squeeze(1))  # (B, 2)
            stop_prob = F.softmax(stop_logit, dim=-1)[:, 1]
            if step >= min_seq_length and (stop_prob > eos_threshold).all():
                break

            # stage / apg: frame-level alpha cosine decay
            cur_alpha_outer = cfg_strength
            if sample_strategy == "stage" and cfg_strength > 1.5:
                cfg_hz = 20
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5
            elif sample_strategy == "apg" and cfg_strength > 1.5:
                cfg_hz = 10
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5

            # history vae window
            y0 = torch.randn(B, self.patch_size, self.audio_channels, device=device, dtype=h_last.dtype)
            t_grid = torch.linspace(0, 1, steps + 1, device=device, dtype=y0.dtype)
            if sway_sampling_coef is not None:
                t_grid = t_grid + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_grid) - 1 + t_grid)

            if vae_results.numel() == 0:
                historical_patch = prompt_vae[:, -self.LocDiT_cfg.history_vae_window_size:, :]
            else:
                current_length = vae_results.shape[1]
                if current_length < self.LocDiT_cfg.history_vae_window_size:
                    historical_patch = torch.cat([prompt_vae[:, -(self.LocDiT_cfg.history_vae_window_size - current_length):, :], vae_results],dim=1,)
                else:
                    historical_patch = vae_results[:, -self.LocDiT_cfg.history_vae_window_size:, :]

            historical_patch_emb = self.hist_to_locdit(historical_patch)
            ctx = self.ctx_to_locdit(h_last)              # (B, 1, D_model)

            cond_locdit_mask = torch.cat([
                torch.ones((B, 1), dtype=torch.bool, device=device),
                torch.ones((B, self.LocDiT_cfg.history_vae_window_size), dtype=torch.bool, device=device),
                torch.ones((B, self.patch_size), dtype=torch.bool, device=device),
            ], dim=1)
            uncond_locdit_mask = cond_locdit_mask  # same shape

            def fn(t, x):
                time_embed = self.time_embedding(t.unsqueeze(0))
                x_emb = self.noisy_to_locdit(x)

                cond_input = torch.cat([ctx, historical_patch_emb, x_emb], dim=1)
                pred = self.LocDiT(
                    x = cond_input,
                    t = time_embed,
                    mask = cond_locdit_mask,
                    patch_size = self.patch_size,
                )

                if cur_alpha_outer < 1e-5:
                    return pred

                # stage schedule along t
                if sample_strategy == "stage":
                    stage_pre = 0.3
                    stage_suf = 0.7
                    if t <= stage_pre:
                        cur_alpha_inner = (cur_alpha_outer - 1.0) * t / stage_pre + 1.0
                    elif t >= stage_suf:
                        cur_alpha_inner = 1.0
                    else:
                        cur_alpha_inner = 0.2
                else:
                    cur_alpha_inner = cur_alpha_outer

                # Training only drops ctx (drop_cond_prob acts on ctx), so uncond zeros out only ctx.
                uncond_ctx = torch.zeros_like(ctx)
                uncond_input = torch.cat([uncond_ctx, historical_patch_emb, x_emb], dim=1)
                null_pred = self.LocDiT(
                    x = uncond_input,
                    t = time_embed,
                    mask = uncond_locdit_mask,
                    patch_size = self.patch_size,
                )
                return pred + (pred - null_pred) * cur_alpha_inner

            trajectory = odeint(fn, y0, t_grid, **self.LocDiT_cfg.odeint_kwargs)
            sampled = trajectory[-1]                      # (B, patch_size, audio_channels)
            vae_results = torch.cat((vae_results, sampled), dim=1)

            # Encode the freshly generated patch back to AR input (uses aggregation_inp_linear, like get_ar_input).
            input_vae_patch_emb = self.aggregation_inp_linear(sampled)
            vae_patch_mask = torch.ones((B, self.patch_size), dtype=torch.bool, device=device)
            if self.patch_size == 1:
                aggregation_emb = input_vae_patch_emb
            else:
                aggregation_emb, _ = self.aggregation_encoder(input_vae_patch_emb, padding_mask=vae_patch_mask)  # (B, 1, D_model)
            # Update AR input: kv_cache appends 1 new patch; otherwise accumulate and re-run the whole sequence.
            if kv_cache:
                cur_inputs_embed = aggregation_emb
            else:
                cur_inputs_embed = torch.cat([cur_inputs_embed, aggregation_emb], dim=1)
            cur_padding_mask = torch.cat([cur_padding_mask, torch.ones((B, 1), device=device, dtype=torch.bool)], dim=1)
            cur_modality_type_ids = torch.cat([cur_modality_type_ids, torch.ones((B, 1), dtype=torch.int64, device=device)], dim=1)

        return vae_results, prompt_vae

    @torch.no_grad()
    def sample_batch(
        self,
        prompt_wav,                # (1, T_audio, 1) — single prompt
        prompt_wav_mask,           # (1, T_audio)    bool, true=valid
        text,                      # list[str] of length 1
        text_lengths,              # (1,) int tensor
        text_padding_mask=None,
        n_rollout: int = 4,
        max_seq_length: int = 155, # 25s
        min_seq_length: int = 10,  # 1s
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef=None,
        seed: int = 666,
        sample_strategy: Literal["base", "stage", "apg"] = "stage",
        eos_threshold: float = 0.5,
    ):
        """Same-prompt, batched ``n_rollout`` sampling (ORW rollout).

        Equivalent to running ``self.sample(...)`` ``n_rollout`` times with different noise,
        but as one batched forward (batch dim = ``n_rollout``). Main ORW rollout speedup.
        EOS stops per sample via ``eos_head`` softmax[:, 1]; uncond zeros out only ctx;
        ``active_idx`` drops a sample from the active batch once it stops.

        Returns
        -------
        gen_latents_list : list of ``n_rollout`` tensors ``(1, T_i, audio_channels)``.
        prompt_vae : ``(1, T_prompt, audio_channels)`` — encoded prompt.
        """
        torch.manual_seed(seed)
        self.eval()
        device = prompt_wav.device

        # 1) Encode prompt once (same path as sample()).
        if self.mel_spec_type == "semanticvae":
            prompt_wav = prompt_wav.squeeze(-1).unsqueeze(1)  # (B, 1, T)
            if prompt_wav.dim() == 4:
                prompt_wav = prompt_wav.squeeze(1)
            prompt_vae, prompt_vae_mask, _ = extract_vae_features(self.generator, prompt_wav, prompt_wav_mask)
        else:
            raise NotImplementedError(f"sample_batch does not support mel_spec_type={self.mel_spec_type}")

        # 2) Replicate prompt + text to n_rollout along batch. .contiguous() copies for
        #    later active_idx advanced indexing.
        B = n_rollout
        prompt_vae = prompt_vae.expand(B, -1, -1).contiguous()            # (B, T_prompt, D_vae)
        prompt_vae_mask = prompt_vae_mask.expand(B, -1).contiguous()       # (B, T_prompt)

        rep_text = [text[0] for _ in range(B)]
        rep_text_lengths = text_lengths.repeat(B)                          # (B,)
        text_max_length = rep_text_lengths.max()

        # 3) Initial AR input — batched; same logic as sample().
        (
            ar_inputs_embed,
            ar_padding_mask,
            text_emb,
            bpe_padding_mask,
            vae_features,
            vae_padding_mask,
            vae_projected,
            vae_aggregation,
            vae_aggregation_mask,
            modality_type_ids,
        ) = self.get_ar_input(
            text=rep_text,
            text_max_length=text_max_length,
            vae_features=prompt_vae,
            vae_mask=prompt_vae_mask,
        )
        text_max_length = bpe_padding_mask.shape[1]

        # 4) AR loop, stop per sample + drop finished samples.
        vae_results_per_sample: list = [None for _ in range(B)]
        active_idx = torch.arange(B, device=device)

        cur_inputs_embed = ar_inputs_embed
        cur_padding_mask = ar_padding_mask
        cur_modality_type_ids = modality_type_ids
        vae_results = torch.empty(
            (B, 0, self.audio_channels), dtype=ar_inputs_embed.dtype, device=device
        )

        for step in range(max_seq_length):
            TB = cur_inputs_embed.shape[0]
            if TB == 0:
                break  # all samples stopped

            h_predict, _ = self.causalAR.inference(
                inputs_embed=cur_inputs_embed,
                padding_mask=cur_padding_mask,
                modality_type_ids=cur_modality_type_ids,
            )

            # h_last: (TB, 1, D_model) — this patch's AR ctx.
            h_last = h_predict[:, -1, :].unsqueeze(1)

            # ---- EOS check (eos_head: 2-class CE, take softmax[:, 1]) ----
            stop_logit = self.eos_head(h_last.squeeze(1))           # (TB, 2)
            stop_prob = F.softmax(stop_logit, dim=-1)[:, 1]          # (TB,)

            # ---- stage / apg: frame-level alpha cosine decay (same as sample()) ----
            cur_alpha_outer = cfg_strength
            if sample_strategy == "stage" and cfg_strength > 1.5:
                cfg_hz = 20
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5
            elif sample_strategy == "apg" and cfg_strength > 1.5:
                cfg_hz = 10
                cur_alpha_outer = (cfg_strength - 1.5) * abs(math.cos(math.pi / (2 * cfg_hz) * step)) + 1.5

            # Independent randn noise per active sample (batch dim = TB).
            y0 = torch.randn(TB, self.patch_size, self.audio_channels, device=device, dtype=h_last.dtype)
            t_grid = torch.linspace(0, 1, steps + 1, device=device, dtype=y0.dtype)
            if sway_sampling_coef is not None:
                t_grid = t_grid + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_grid) - 1 + t_grid)

            # History vae window per active sample: active subset of (replicated) prompt + running vae_results.
            if vae_results.numel() == 0:
                historical_patch = prompt_vae[active_idx, -self.LocDiT_cfg.history_vae_window_size:, :]
            else:
                current_length = vae_results.shape[1]
                if current_length < self.LocDiT_cfg.history_vae_window_size:
                    historical_patch = torch.cat(
                        [prompt_vae[active_idx, -(self.LocDiT_cfg.history_vae_window_size - current_length):, :], vae_results],
                        dim=1,
                    )
                else:
                    historical_patch = vae_results[:, -self.LocDiT_cfg.history_vae_window_size:, :]

            historical_patch_emb = self.hist_to_locdit(historical_patch)
            ctx = self.ctx_to_locdit(h_last)                         # (TB, 1, D_model)

            cond_locdit_mask = torch.cat([
                torch.ones((TB, 1), dtype=torch.bool, device=device),
                torch.ones((TB, self.LocDiT_cfg.history_vae_window_size), dtype=torch.bool, device=device),
                torch.ones((TB, self.patch_size), dtype=torch.bool, device=device),
            ], dim=1)
            uncond_locdit_mask = cond_locdit_mask  # same shape

            def fn(t, x):
                # LocDiT folds/unfolds the patch dim with batch_size = t.shape[0], so for
                # TB>1 time must carry batch dim TB or the output reshape misaligns.
                time = t.unsqueeze(0).expand(x.shape[0])             # (TB,)
                time_embed = self.time_embedding(time)               # (TB, D_model)
                x_emb = self.noisy_to_locdit(x)
                cond_input = torch.cat([ctx, historical_patch_emb, x_emb], dim=1)
                pred = self.LocDiT(
                    x=cond_input,
                    t=time_embed,
                    mask=cond_locdit_mask,
                    patch_size=self.patch_size,
                )

                if cur_alpha_outer < 1e-5:
                    return pred

                # stage schedule along t (same as sample()).
                if sample_strategy == "stage":
                    stage_pre = 0.3
                    stage_suf = 0.7
                    if t <= stage_pre:
                        cur_alpha_inner = (cur_alpha_outer - 1.0) * t / stage_pre + 1.0
                    elif t >= stage_suf:
                        cur_alpha_inner = 1.0
                    else:
                        cur_alpha_inner = 0.2
                else:
                    cur_alpha_inner = cur_alpha_outer

                # Training only drops ctx (drop_cond_prob acts on ctx), so uncond zeros out only ctx.
                uncond_ctx = torch.zeros_like(ctx)
                uncond_input = torch.cat([uncond_ctx, historical_patch_emb, x_emb], dim=1)
                null_pred = self.LocDiT(
                    x=uncond_input,
                    t=time_embed,
                    mask=uncond_locdit_mask,
                    patch_size=self.patch_size,
                )
                return pred + (pred - null_pred) * cur_alpha_inner

            trajectory = odeint(fn, y0, t_grid, **self.LocDiT_cfg.odeint_kwargs)
            sampled = trajectory[-1]                                 # (TB, patch_size, audio_channels)
            vae_results = torch.cat((vae_results, sampled), dim=1)

            # ----- stop per sample -----
            if step >= min_seq_length:
                stop_now = stop_prob > eos_threshold                 # (TB,) bool
            else:
                stop_now = torch.zeros(TB, dtype=torch.bool, device=device)

            if stop_now.any():
                # Snapshot stopped samples by their original index into vae_results_per_sample.
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
                    break  # all stopped this step

            # ----- encode the freshly generated patch back to AR input for the next step -----
            input_vae_patch_emb = self.aggregation_inp_linear(sampled)
            vae_patch_mask = torch.ones(
                (sampled.shape[0], self.patch_size), dtype=torch.bool, device=device,
            )
            if self.patch_size == 1:
                aggregation_emb = input_vae_patch_emb
            else:
                aggregation_emb, _ = self.aggregation_encoder(
                    input_vae_patch_emb, padding_mask=vae_patch_mask
                )                                                    # (TB_active, 1, D_model)

            cur_inputs_embed = torch.cat([cur_inputs_embed, aggregation_emb], dim=1)
            cur_padding_mask = torch.cat(
                [cur_padding_mask, torch.ones((cur_inputs_embed.shape[0], 1), device=device, dtype=torch.bool)],
                dim=1,
            )
            cur_modality_type_ids = torch.cat(
                [cur_modality_type_ids, torch.ones((cur_inputs_embed.shape[0], 1), dtype=torch.int64, device=device)],
                dim=1,
            )

        # Snapshot samples that never stopped within max_seq_length.
        for i_local, orig_i in enumerate(active_idx.tolist()):
            if vae_results_per_sample[orig_i] is None:
                vae_results_per_sample[orig_i] = vae_results[i_local].clone()

        # Assemble output as list[(1, T_i, audio_channels)], matching the n_rollout for-loop output.
        gen_latents_list = [v.unsqueeze(0) for v in vae_results_per_sample]

        return gen_latents_list, prompt_vae[:1]

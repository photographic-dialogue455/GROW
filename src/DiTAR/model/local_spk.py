from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import torchaudio

# The WavLM ECAPA-TDNN encoder (eval ``run_sim``) also operates at 16 kHz.
_WAVLM_SR = 16000
# Default fine-tuned WavLM weights, matching DiTAR.eval.eval_librispeech_test_clean.
_WAVLM_DEFAULT_CKPT = "checkpoints/wavlm_large_finetune.pth"


# ---------------------------------------------------------------------------
# Shared resampler cache (cheap torchaudio transforms; not a model download)
# ---------------------------------------------------------------------------

_RESAMPLER_CACHE: dict = {}


def _get_resampler(src_sr: int, dst_sr: int, device) -> torchaudio.transforms.Resample:
    key = (int(src_sr), int(dst_sr), str(device))
    if key not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(
            orig_freq=src_sr, new_freq=dst_sr
        ).to(device)
    return _RESAMPLER_CACHE[key]


# ---------------------------------------------------------------------------
# Eager WavLM ECAPA-TDNN loader (matches DiTAR.eval.utils_eval.run_sim)
# ---------------------------------------------------------------------------


def build_wavlm_spk_model(device, ckpt_path: str):
    from DiTAR.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

    ckpt_path = ckpt_path or _WAVLM_DEFAULT_CKPT
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_mono_1d(a: torch.Tensor) -> torch.Tensor:
    """Squeeze a (1,1,T) / (1,T) / (T,1) / (T,) waveform tensor down to (T,)."""
    a = a.squeeze()
    if a.dim() == 0:  # degenerate single-sample
        a = a.view(1)
    elif a.dim() > 1:
        # Fall back: collapse leading dims, keep the longest axis as time.
        a = a.reshape(-1)
    return a


# ---------------------------------------------------------------------------
# Public batch entry point — WavLM ECAPA-TDNN (matches eval run_sim)
# ---------------------------------------------------------------------------


def _wav_to_16k_tensor(a: torch.Tensor, source_sr: int, device) -> torch.Tensor:
    """(any-shape) torch waveform at ``source_sr`` -> (1, T) float32 @ 16 kHz on ``device``.

    Mirrors the per-clip preprocessing in ``DiTAR.eval.utils_eval.run_sim``:
    resample to 16 kHz and feed a (1, T) tensor to the encoder.
    """
    a = _to_mono_1d(a).float().to(device)
    if int(source_sr) != _WAVLM_SR:
        resampler = _get_resampler(source_sr, _WAVLM_SR, device)
        a = resampler(a)
    return a.unsqueeze(0)  # (1, T)


@torch.no_grad()
def run_wavlm_sim_local_batch(
    audios: List[torch.Tensor],
    prompt_audios: List[torch.Tensor],
    gen_source_sr: int = 16000,
    prompt_source_sr: int = 16000,
    device=None,
    model=None,
) -> List[dict]:
    """Score each generated waveform by WavLM speaker similarity to its prompt.

    In-memory counterpart of ``DiTAR.eval.utils_eval.run_sim``: uses the same
    ``ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")`` encoder with the
    fine-tuned ``wavlm_large_finetune.pth`` weights, and scores by the cosine
    similarity of the 256-d embeddings (range roughly ``[-1, 1]``).

    Args:
        audios: list of generated waveforms (the ORW rollouts). Each is a
            1-D / (1, T) / (1, 1, T) tensor at ``gen_source_sr``.
        prompt_audios: list of *reference-speaker* waveforms aligned with
            ``audios`` (same length). For single-prompt ORW the caller passes
            the same prompt tensor ``n_rollout`` times — we embed each unique
            prompt object only once.
        gen_source_sr: sample rate of ``audios``.
        prompt_source_sr: sample rate of ``prompt_audios``.
        device: device for the encoder / resamplers; defaults to the device of
            the first generated audio.
        model: a pre-built WavLM ECAPA-TDNN encoder (see ``build_wavlm_spk_model``);
            it MUST be supplied by the caller (the ORW module builds it once at
            training start and passes it in).

    Returns:
        list of ``{"score": sim, "sim": sim}`` dicts, one per ``audios`` entry.
    """
    if len(audios) == 0:
        return []
    assert len(audios) == len(prompt_audios), (
        f"audios ({len(audios)}) and prompt_audios ({len(prompt_audios)}) length mismatch"
    )

    if device is None:
        device = audios[0].device

    # Embed each *unique* prompt only once (single-prompt ORW reuses one tensor).
    prompt_embed_cache: dict = {}

    def _prompt_embed(p: torch.Tensor):
        key = id(p)
        if key not in prompt_embed_cache:
            wav = _wav_to_16k_tensor(p, prompt_source_sr, device)
            prompt_embed_cache[key] = model(wav)
        return prompt_embed_cache[key]

    out: List[dict] = []
    for gen_a, prompt_a in zip(audios, prompt_audios):
        p_emb = _prompt_embed(prompt_a) #torch.Size([1, 256])
        g_emb = model(_wav_to_16k_tensor(gen_a, gen_source_sr, device)) #torch.Size([1, 256])
        sim = float(F.cosine_similarity(g_emb, p_emb)[0].item())

        out.append({"score": sim, "sim": sim})

    return out

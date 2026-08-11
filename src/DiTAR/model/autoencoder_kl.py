import json
from pathlib import Path

import torch

from DiTAR.vae_dac.model import DAC


def make_pad_mask(lengths, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    if type(lengths) is list:
        lengths = torch.tensor(lengths)
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(-1)


@torch.no_grad()
def process_online(signal, generator):
    audio_data = generator.preprocess(signal, generator.sample_rate)
    latent, mu, log_var, kl_loss = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu, log_var)
    return pre_proj_latent.transpose(1, 2)


def read_json_file(metainfo_path):
    with open(metainfo_path, "r") as f:
        return json.load(f)


def load_state(folder: str):
    print(f"Resuming from {str(Path('.').absolute())}/{folder}")
    metainfo = read_json_file(Path(".").absolute() / folder / "metainfo.json")
    ckpt_path = Path(folder) / "dac" / "ema_state_dict.pth"
    model_dict = torch.load(ckpt_path, map_location="cpu")
    ckpt_dict = {k.replace("ema_model.", ""): v for k, v in model_dict.items()}
    filter_dict = {k: v for k, v in ckpt_dict.items() if not k.startswith("projectors")}
    print(f"Load from {ckpt_path}")
    generator = DAC(**metainfo["DAC"])
    del generator.projectors
    generator.load_state_dict(filter_dict, strict=False)
    generator.eval()
    return generator


def build_generator(path=None):
    generator = load_state(folder=path)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    return generator


@torch.no_grad()
def extract_vae_features(generator, vae_features, vae_padding_mask):
    vae_features = process_online(vae_features, generator).transpose(1, 2).detach()
    original_valid_lengths = vae_padding_mask.sum(dim=1)
    valid_feature_lengths = torch.ceil(original_valid_lengths.float() / generator.hop_length).long()
    vae_padding_mask = ~make_pad_mask(valid_feature_lengths, max_len=valid_feature_lengths.max().item())  # B t
    return vae_features, vae_padding_mask, valid_feature_lengths

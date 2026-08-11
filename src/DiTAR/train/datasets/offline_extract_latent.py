import argbind
import torch
from pathlib import Path
from DiTAR.vae_dac.model import DAC
import numpy as np
from tqdm import tqdm
import json
import torchaudio
import glob


def find_audio(folder: str, ext=[".wav", ".flac", ".mp3", ".mp4"]):
    folder = Path(folder)
    if str(folder).endswith(tuple(ext)):
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files


def read_json_file(metainfo_path):
    with open(metainfo_path, "r") as f:
        data = json.load(f)
    return data


def load_state(
    save_path: str,
    tag: str = "latest",
    use_ema: bool = True
):
    kwargs = {
        "folder": f"{save_path}/{tag}",
        "map_location": "cpu",
    }
    print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
    metainfo_path = Path(".").absolute() / kwargs["folder"] / "metainfo.json"
    metainfo = read_json_file(metainfo_path)
    if not use_ema:
        ckpt_path = Path(kwargs["folder"]) / "dac" / "weights.pth"
        model_dict = torch.load(ckpt_path, map_location=kwargs["map_location"])
        filter_dict = {
            k: v
            for k, v in model_dict["state_dict"].items()
            if not k.startswith("projectors") and not k.startswith("decoder")
        }
    else:
        ckpt_path = Path(kwargs["folder"]) / "dac" / "ema_state_dict.pth"
        model_dict = torch.load(ckpt_path, map_location=kwargs["map_location"])
        ckpt_dict = {k.replace("ema_model.",""):v for k,v in model_dict.items()}
        filter_dict = {
            k: v
            for k, v in ckpt_dict.items()
            if not k.startswith("projectors") and not k.startswith("decoder")
        }
        print(f"Load from {ckpt_path}, use_ema: {use_ema}")
    decoder_dict = {
        "resblock": "1",
        "num_mels": 64,
        "upsample_rates": [5, 5, 2, 2, 2, 2],
        "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4],
        "upsample_initial_channel": 1024,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "use_tanh_at_final": False,
        "use_bias_at_final": False,
        "activation": "snakebeta",
        "snake_logscale": True,
    }

    generator = DAC(**metainfo["DAC"], decoder_dict=decoder_dict)
    del generator.projectors
    del generator.decoder
    del generator.decoder_proj
    print(generator)
    msg = generator.load_state_dict(filter_dict, strict=False)
    print(msg)
    generator.eval()
    return generator


@torch.no_grad()
def process(signal, generator, **kwargs):
    audio_data = generator.preprocess(signal, generator.sample_rate)
    latent, mu, log_var, kl_loss = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu, log_var)
    return pre_proj_latent.transpose(1, 2).cpu().numpy()

@torch.no_grad()
def process_online(signal, generator, **kwargs):  # encoder only
    audio_data = generator.preprocess(signal, generator.sample_rate)
    latent, mu, log_var, kl_loss = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu, log_var)
    return pre_proj_latent.transpose(1, 2)


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    path: str = "ckpt",
    input: str = "samples/input",
    output: str = "samples/output",
    model_tag: str = "best",
):
    generator = load_state(
        save_path=path,
        tag=model_tag,
    ).cuda()
    generator.eval()

    audio_files = find_audio(input)
    print(f"Audio Nums = {len(audio_files)}")
    print(f"Generator SR = {generator.sample_rate}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(audio_files))):
        relative_path = audio_files[i].relative_to(input)
        output_path = output / relative_path
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if output_path.with_suffix(".npy").exists():
            continue
        signal, sample_rate = torchaudio.load(str(audio_files[i]))
        signal = signal.unsqueeze(0).cuda()
        if sample_rate != generator.sample_rate:
            signal = torchaudio.functional.resample(signal, orig_freq=sample_rate, new_freq=generator.sample_rate)
        feat = process(signal, generator)  # 1 dim seq_len
        np.save(output_path.with_suffix(".npy"), feat)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()

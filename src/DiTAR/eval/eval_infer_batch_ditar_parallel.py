import os
import sys

sys.path.append(os.getcwd())

import argparse
import time

import torch
import torchaudio
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm

from DiTAR.eval.utils_eval import (
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from DiTAR.infer.utils_infer import load_checkpoint, load_vocoder
from DiTAR.model.ldm_dataset import CustomLDMEvalDataset
from DiTAR.model import DiTAR
from DiTAR.train.train_ditar import create_model

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1

# rel_path = str(files("DiTAR").joinpath("../../"))
rel_path = "."
def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, )

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument("-p", "--prompt_vae_path", default="")
    parser.add_argument("--cfg_scale", default=2.0, type=float)

    parser.add_argument("--config_path", default="", type=str)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--metalst", default="", type=str)
    parser.add_argument("--kv_cache", action='store_true')
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--sample_strategy", default="stage", type=str,choices=["base", "stage", "apg"])
    # Optional explicit checkpoint path. If given, overrides the default
    # `./ckpts/{expname}/model_{ckptstep}.pt` layout so released checkpoints can
    # live anywhere (e.g. model_ckpts/03_grow_nfe10/model_750.pt).
    parser.add_argument("--ckpt_path", default="", type=str)

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset
    prompt_vae_path = args.prompt_vae_path

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = args.cfg_scale

    config_path = args.config_path
    tag = args.tag
    metalst = args.metalst
    kv_cache = args.kv_cache
    output_dir_arg = args.output_dir
    sample_strategy = args.sample_strategy

    cfg = OmegaConf.load(config_path)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone.pretrained_LM_path)

    mel_spec_type = cfg.model.vocoder.mel_spec_type
    if mel_spec_type == "semanticvae":
        target_sample_rate = 16000
    else:
        raise ValueError(f"Unsupported mel_spec_type: {mel_spec_type}")
        
    if testset == "ls_pc_test_clean":
        # LibriSpeech-PC test-clean dir. The eval dataset rebuilds each flac path
        # from `-p/{spk}/{chapter}/{utt}.flac`, so honor `-p` (--prompt_vae_path)
        # and only fall back to a hardcoded path if it is not given.
        librispeech_test_clean_path = prompt_vae_path or "/path/to/LibriSpeech/test-clean"
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        # honor --metalst override (e.g. meta_debug.lst); fallback to full meta.lst
        if not metalst:
            metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        # honor --metalst override (e.g. meta_debug.lst); fallback to full meta.lst
        if not metalst:
            metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    if output_dir_arg:
        output_dir = output_dir_arg
    else:
        output_dir = (
            f"{rel_path}/"
            f"results/{exp_name}_{ckpt_step}/{testset}/"
            f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
            f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
            f"_cfg{cfg_strength}_bsz{infer_batch_size}"
            f"_{tag}"
            f"_cross_sentence"
        )
    print(output_dir)

    # -------------------------------------------------#

    dataset_test = CustomLDMEvalDataset(
        metainfo,
        prompt_vae_path,
        testset_name=testset,
        audio_type=cfg.model.audio_type,
        target_sample_rate = target_sample_rate,
        pad_multiple_of = None,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers = cfg.datasets.num_workers,
            pin_memory=True,
            shuffle=False,
            batch_size=infer_batch_size, 
            drop_last=False,
            collate_fn=dataset_test.collate_fn
        )

    test_dataloader = accelerator.prepare(test_dataloader)

    if mel_spec_type == "semanticvae":
        vocoder = load_vocoder(
            vocoder_name=mel_spec_type,
            is_local=cfg.model.vocoder.is_local,
            local_path=cfg.model.vocoder.local_path,
        )
        # ensure the vocoder is on this process's GPU

    vocoder = vocoder.to(device)
        
    model_name = str(cfg.model.name)
    ckpt_path = args.ckpt_path if args.ckpt_path else rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.pt"
    print(f"Loading from {ckpt_path}  (cfg.model.name={model_name})")
    dtype = torch.float32

    if model_name in ("DiTAR_ORW", "DiTAR_SDE"):
        model = DiTAR(ditar_cfg=cfg.model, tokenizer=tokenizer).to(device)
        model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=False)
    else:
        model = create_model(cfg.model, tokenizer).to(device)
        model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)


    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    start = time.time()

    for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        for key in batch.keys():
            batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
        key = batch["key"]
    
        output_filepath = f"{output_dir}/{key[0]}.wav"
        if os.path.exists(output_filepath):
            print(f"skipping {output_filepath}")
            continue

        prompt_vae = batch["prompt_vae"]
        vae_mask = batch["vae_mask"]
        phn = batch["phn"]
        phn_lengths = batch["phn_lengths"]
        gt_text = batch["gt_text"]

        # Inference
        with torch.inference_mode():
            sampled, _ = model.sample(
                prompt_vae,
                vae_mask,
                phn,
                phn_lengths,
                steps=nfe_step, #default 32
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, # default -1
                seed = seed,
                sample_strategy = sample_strategy,
                kv_cache = kv_cache
            )
            gen_mel_spec = sampled  
            
            if cfg.model.vocoder.mel_spec_type == "semanticvae":
                gen_audio = vocoder.decode(gen_mel_spec).squeeze(0).cpu()

            torchaudio.save(
                f"{output_dir}/{key[0]}.wav", gen_audio, target_sample_rate
            )
            
            if accelerator.is_local_main_process:
                print(f"saving audio {output_dir}/{key[0]}.wav")
                if gt_text:
                    print(gt_text)


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
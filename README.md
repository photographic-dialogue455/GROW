# GROW: Group-Relative Advantage-Weighted On-Policy Reinforcement Learning of Autoregressive-Diffusion Text-to-Speech Model

[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-Paper-red.svg)](https://arxiv.org/abs/2608.03215)
[![github](https://img.shields.io/badge/Code-Repo-black?logo=github)](https://github.com/yanghaha0908/GROW)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20GROW-Models-blueviolet)](https://huggingface.co/yhaha/GROW)

GROW is a group-relative advantage-weighted on-policy reinforcement learning method
for flow-matching text-to-speech. It acts directly on the standard flow-matching
objective: for each prompt it samples a group of on-policy utterances, standardizes
intelligibility and speaker-similarity rewards within the group, and combines them to
reweight the flow-matching regression, with a Wasserstein-2 velocity penalty anchoring
the model to a frozen pretrained reference. Instantiated on **DiTAR** and evaluated on
LibriSpeech and Seed-TTS EN/ZH, GROW reduces average WER from 2.016 to 1.558 and raises
speaker similarity from 0.676 to 0.715 while keeping UTMOS.

This is the official code for the paper [*GROW: Group-Relative Advantage-Weighted
On-Policy Reinforcement Learning of Autoregressive-Diffusion Text-to-Speech Model*](https://arxiv.org/abs/2608.03215).

## ✨ Key Features
- **On-policy RL directly on flow matching** – no ODE→SDE conversion or per-step likelihood-ratio tracking.
- **Group-relative advantage weighting** – within-group standardized rewards with a group-mean baseline for effective credit assignment.
- **Efficient** – with 10-NFE training rollouts and 32-NFE evaluation, GROW trains 2.9× faster than 32-NFE DiTAR-GRPO.

## 🛠️ Installation

We recommend a fresh conda environment. (Tested on Python 3.10, NVIDIA A800-80GB,
CUDA 12.6, and PyTorch 2.7.0.)

### 1. System packages
Audio I/O needs **ffmpeg / sox / libsndfile**:
```bash
# Debian / Ubuntu
sudo apt-get install -y ffmpeg sox libsndfile1
# CentOS
sudo yum install -y ffmpeg sox libsndfile
```

### 2. Environment
```bash
conda create -n grow python=3.10 -y
conda activate grow
```

### 3. Install
```bash
git clone https://github.com/yanghaha0908/GROW.git
cd GROW

# PyTorch (CUDA 12.6 build)
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126

# remaining dependencies
pip install -r requirements.txt
pip install -e .
```

Everything above is also bundled in `install.sh`.



## 📦 Pretrained Checkpoints

The three released checkpoints correspond to rows of **Table 1** in the paper and
live on 🤗 [**yhaha/GROW**](https://huggingface.co/yhaha/GROW).

| Folder | Table 1 row | Model class | Checkpoint | Notes |
|--------|-------------|-------------|-----------|-------|
| `01_pretrain/`         | Row 1 | `DiTAR`     | `model_200000.pt` | 200K-step pretrain; also the frozen reference for RL |
| `02_ditar_grpo_nfe10/` | Row 3 | `DiTAR_SDE` | `model_750.pt`    | Flow-GRPO baseline, rollout NFE=10, β<sub>W2</sub>=0, lr=1e-5 |
| `03_grow_nfe10/`       | Row 6 | `DiTAR_ORW` | `model_750.pt`    | **Default GROW setting**: rollout NFE=10, β<sub>W2</sub>=0.025, lr=2e-6 |

Reported full-test-set metrics (WER↓ / SIM↑ — LibriSpeech-PC · Seed-TTS EN · Seed-TTS ZH):

| Checkpoint | LS WER/SIM | Seed-EN WER/SIM | Seed-ZH WER/SIM |
|------------|-----------|-----------------|-----------------|
| `01_pretrain`         | 2.373 / 0.648 | 2.406 / 0.663 | 1.269 / 0.717 |
| `02_ditar_grpo_nfe10` | 2.332 / 0.683 | 1.728 / 0.699 | 1.180 / 0.738 |
| `03_grow_nfe10`       | 1.927 / 0.701 | 1.763 / 0.702 | 0.983 / 0.742 |

Download all three folders (each folder ships
its `config.yaml` next to the `.pt` weights):
```bash
huggingface-cli download yhaha/GROW --local-dir model_ckpts
```



Beyond the GROW weights, we need two external models under `checkpoints/`:

- **`checkpoints/Qwen3-0.6B/`** — AR backbone LM from
  🤗 [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).
- **`checkpoints/Semantic-VAE/semantic_vae_1000k/`** — the DiTAR speech autoencoder: a
  16 kHz DAC-style encoder with a 64-dim VAE bottleneck (the latent space DiTAR models) and
  a BigVGAN decoder that vocodes latents back to waveform, from
  [`ZhikangNiu/Semantic-VAE`](https://github.com/ZhikangNiu/Semantic-VAE).
 

The ASR / speaker / UTMOS models used only for **evaluation and RL rewards** are listed in [`src/DiTAR/eval/README.md`](src/DiTAR/eval/README.md).

## 📁 Data Preparation

GROW uses a two-stage data pipeline: standard TTS **pretraining** data, then a
small **RL** dataset for the reinforcement-learning stage.

### Pretraining data

DiTAR pretraining reuses the **F5-TTS** Emilia pipeline, then does one extra format
conversion.

**Step 1 — F5-TTS Emilia preparation.** Run the upstream script
[`prepare_emilia_v2.py`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/datasets/prepare_emilia_v2.py):

```bash
# Edit dataset_dir / dataset_name at the bottom of the upstream script, then:
python prepare_emilia_v2.py
```

It writes a single `raw.arrow` (via `datasets`' `ArrowWriter`), a `duration.json`, and a
`vocab.txt`, with the schema `{audio_path, text, duration}`.

**Step 2 — convert to the `save_to_disk` layout.** GROW's loader
(`DiTAR.model.ldm_dataset.load_ldm_dataset`) uses `datasets.load_from_disk`, which expects
the `save_to_disk` directory format (`data-*.arrow` + `dataset_info.json` + `state.json`),
**not** a bare `raw.arrow`. Convert it with
[`src/DiTAR/train/datasets/convert_raw_arrow_to_save_to_disk.py`](src/DiTAR/train/datasets/convert_raw_arrow_to_save_to_disk.py):

```bash
python src/DiTAR/train/datasets/convert_raw_arrow_to_save_to_disk.py \
    --src data/Emilia_ZH_EN_char_raw \
    --dst data/Emilia_ZH_EN_char
```

This re-saves the same `{audio_path, text, duration}` data (and carries over
`duration.json`). Point the pretraining config's `datasets.train_ds_path` at the `--dst`
directory.

### RL data

The RL fine-tuning dataset is built by
[`src/DiTAR/train/datasets/build_rl_mixed_10k.py`](src/DiTAR/train/datasets/build_rl_mixed_10k.py).
It samples prompts from Emilia (EN + ZH) and LibriTTS `train-clean-100` (EN), and for
each prompt assigns a `gen_text` drawn from a real recording in the same corpus/language
group:

```bash
# full build → data/RL_mixed_emilia_libritts_30k (30k rows: Emilia-EN 10k + Emilia-ZH 10k + LibriTTS-EN 10k)
python src/DiTAR/train/datasets/build_rl_mixed_10k.py \
    --emilia_src   data/Emilia_ZH_EN_char \
    --libritts_src data/LibriTTS/train-clean-100 \
    --out          data/RL_mixed_emilia_libritts_30k
```

The output has columns `{audio_path, text, gen_text, duration, language}` plus a
matching `duration.json`. Point the RL config's `datasets.train_ds_path` at
`data/RL_mixed_emilia_libritts_30k`.

## 🔧 Training

Each script writes its
checkpoints and `train.log` under `ckpts/<exp_name>/`.

### 1. Pretraining (DiTAR)
```bash
bash run_0.6B_1e-4_pray_qwen_emilia_semanticvae_again.sh
```

### 2. RL fine-tuning
Both RL scripts initialize policy **and** the frozen reference from the pretrained
checkpoint `model_ckpts/01_pretrain/model_200000.pt` (set via `model.orw.ref_ckpt_path`).

```bash
# GROW (advantage-weighted flow-matching RL) — paper default
bash run_ditar_orw_sim_and_wer_step10_beta0.025_lr2e-6.sh

# DiTAR-GRPO baseline (SDE / Flow-GRPO)
bash run_ditar_sde_sim_and_wer_step10_beta0lr1e-5.sh
```

Key RL knobs (overridable as Hydra `++model.orw.*` args): `n_rollout` (group size),
`beta` (W2 anchor weight), `rm_method` (`grpo`), `sample_steps` (rollout NFE),
`reward_type` (`sim_and_wer` | `sim` | `wer`), `wer_weight`, `sim_weight`.

## 🚀 Inference & Evaluation

**Evaluation data.** The eval script reads three standard zero-shot TTS test sets under
`data/` (paths overridable via env vars in `eval_8gpu.sh`):

- **LibriSpeech-PC** — `data/LibriSpeech/test-clean/` ([OpenSLR-12](https://www.openslr.org/12))
  plus the cross-sentence manifest `data/librispeech_pc_test_clean_cross_sentence.lst`
  (from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)).
- **Seed-TTS EN / ZH** — `data/seedtts_testset/{en,zh}/` with each split's `meta.lst` +
  prompt wavs ([seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)).

The eval-time ASR / speaker / UTMOS models are listed in
[`src/DiTAR/eval/README.md`](src/DiTAR/eval/README.md).

The 8-GPU evaluation script reproduces the paper's Table 1 (WER / SIM / UTMOS on
LibriSpeech-PC · Seed-TTS EN · Seed-TTS ZH):

```bash
# all three checkpoints × all three test sets
bash eval_8gpu.sh

# a single checkpoint / test set
CKPTS="03_grow_nfe10" bash eval_8gpu.sh
TESTSETS="ls" bash eval_8gpu.sh          # ls | en | zh
```

Each `(checkpoint × test set)` job runs distributed inference, then SIM → UTMOS → WER;
jobs run sequentially, each using all 8 GPUs. The run is resumable — existing
wavs and metric `.jsonl` files are skipped.

## ❤️ Acknowledgements

We sincerely thank the authors of the following open-source projects, whose excellent
work laid the foundation for GROW: [F5-TTS](https://github.com/SWivid/F5-TTS), [Semantic-VAE](https://github.com/ZhikangNiu/Semantic-VAE).

## 📝 Citation

If you find this repo helpful, please cite our work:

```bibtex
@article{yang2026grow,
  title={GROW: Group-Relative Advantage-Weighted On-Policy Reinforcement Learning of Autoregressive-Diffusion Text-to-Speech model},
  author={Yang, Guanrou and Tan, Tian and Chen, Qian and Ma, Ziyang and Song, Yakun and Niu, Zhikang and Chen, Qi and Tu, Wenming and Li, Haitao and Yang, Shan and others},
  journal={arXiv preprint arXiv:2608.03215},
  year={2026}
}
```

## 📄 License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE) for details.

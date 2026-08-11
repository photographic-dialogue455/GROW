# GROW checkpoints

The three checkpoints released here correspond to rows of **Table 1** in the
paper. 

| Folder | Table 1 row | Model | Checkpoint | Notes |
|--------|-------------|-------|-----------|-------|
| `01_pretrain/`         | Row 1 | Pretrained DiTAR (`DiTAR`)        | `model_200000.pt` | 200K-step pretrain; also the frozen reference *q* for RL |
| `02_ditar_grpo_nfe10/` | Row 3 | DiTAR-GRPO, NFE=10 (`DiTAR_SDE`)  | `model_750.pt`    | Flow-GRPO baseline, rollout NFE=10, β<sub>W2</sub>=0, lr=1e-5 |
| `03_grow_nfe10/`       | Row 6 | GROW, NFE=10 (`DiTAR_ORW`)        | `model_750.pt`    | **Default GROW setting**: rollout NFE=10, β<sub>W2</sub>=0.025, lr=2e-6 |

Reported full-test-set metrics (WER / SIM — LibriSpeech-PC · Seed-TTS EN · Seed-TTS ZH):

| Checkpoint | LS WER/SIM | Seed-EN WER/SIM | Seed-ZH WER/SIM |
|------------|-----------|-----------------|-----------------|
| `01_pretrain`         | 2.373 / 0.648 | 2.406 / 0.663 | 1.269 / 0.717 |
| `02_ditar_grpo_nfe10` | 2.332 / 0.683 | 1.728 / 0.699 | 1.180 / 0.738 |
| `03_grow_nfe10`       | 1.927 / 0.701 | 1.763 / 0.702 | 0.983 / 0.742 |

Each folder ships the training/inference `config.yaml` next to the
`.pt` weights. The eval loader reads model weights from `model_state_dict`
(RL checkpoints) or `ema_model_state_dict` (the pretrained checkpoint),
matching `cfg.model.name`.

## Layout

```
model_ckpts/
├── 01_pretrain/          config.yaml   model_200000.pt
├── 02_ditar_grpo_nfe10/  config.yaml   model_750.pt
└── 03_grow_nfe10/        config.yaml   model_750.pt
```

> The `.pt` weights are large (4–13 GB) and are distributed via HuggingFace,
> not git. Download them into the matching folder
> before running evaluation.

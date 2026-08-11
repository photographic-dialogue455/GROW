# Evaluation & RL-reward checkpoints

The metrics (WER / SIM / UTMOS) and the RL rewards need a few extra models under
`checkpoints/`. The left column is the exact path the eval scripts expect.

| `checkpoints/…` | Model | Used for | Source |
|-----------------|-------|----------|--------|
| `wavlm_large_finetune.pth` | WavLM-Large (speaker verification) | SIM | [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) → [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) |
| `faster-whisper-large-v3/` | faster-whisper large-v3 | EN WER | 🤗 [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| `funasr/paraformer-zh/` | FunASR Paraformer-large | ZH WER | 🤗 [funasr/paraformer-zh](https://huggingface.co/funasr/paraformer-zh) |

**UTMOS** is not stored locally — `eval_utmos.py` fetches `utmos22_strong` (tag `v1.2.0`)
via `torch.hub` from [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS).

```bash
pip install -U "huggingface_hub[cli]" gdown
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir checkpoints/faster-whisper-large-v3
huggingface-cli download funasr/paraformer-zh            --local-dir checkpoints/funasr/paraformer-zh
gdown "https://drive.google.com/uc?id=1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP" -O checkpoints/wavlm_large_finetune.pth
```

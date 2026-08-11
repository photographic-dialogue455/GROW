import os
import json
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_from_disk
from torch.utils.data import Dataset

from DiTAR.model.utils import make_pad_mask


# Root of the Emilia-Dataset. Emilia audio paths are reconstructed from the
# clip filename (see CustomLDMDataset.__getitem__). Point this at your local
# Emilia-Dataset copy, or set the EMILIA_DATASET_ROOT environment variable.
EMILIA_DATASET_ROOT = os.environ.get("EMILIA_DATASET_ROOT", "data/Emilia-Dataset")


def detect_lang_from_path(path) -> str:
    """Infer the language ("zh" / "en") from an Emilia-style audio path.

    e.g. '/.../Emilia/ZH/ZH_B00059_S06833_W000015.mp3' -> 'zh'
         '/.../Emilia/EN/EN_B00017_S08979_W000006.mp3' -> 'en'

    The filename prefix (ZH_ / EN_) is checked first, then the directory
    marker (/ZH/ , /EN/). Falls back to 'en' (a safe default for the
    downstream WER text normalization) when nothing matches.
    """
    if isinstance(path, (list, tuple)):
        path = path[0] if len(path) > 0 else ""
    if not path:
        return "en"
    p = str(path)
    base = os.path.basename(p).upper()
    if base.startswith("ZH"):
        return "zh"
    if base.startswith("EN"):
        return "en"
    up = p.upper()
    if "/ZH/" in up or "\\ZH\\" in up:
        return "zh"
    if "/EN/" in up or "\\EN\\" in up:
        return "en"
    return "en"


@dataclass
class LDMDataFilterConfig:
    max_seq_len: int = 30
    min_seq_len: int = 1
    min_dnsmos: float = 2.5
    disabled: bool = False


class CustomLDMDataset(Dataset):
    """Training dataset. Loads raw waveforms online ("vae_online"); the VAE
    encoding is done downstream in the model."""

    def __init__(
        self,
        dataset: Dataset,
        target_sample_rate: int,
        filter_config: LDMDataFilterConfig = None,
        durations=None,
        audio_type: str = "vae_online",
    ):
        if audio_type != "vae_online":
            raise NotImplementedError(
                f"Only 'vae_online' audio_type is supported; got {audio_type!r}"
            )
        self.data = dataset
        self.durations = durations
        self.filter_config = filter_config or LDMDataFilterConfig()
        self.audio_type = audio_type
        self.target_sample_rate = target_sample_rate
        self.pad_multiple_of = 480  # BigVGAN hop size

    def get_frame_len(self, index):
        # Used by DynamicBatchSampler. If durations are provided separately they
        # must be correct, otherwise batching will almost certainly OOM.
        if self.durations is not None:
            return self.durations[index]
        return self.data[index]["duration"]

    def __len__(self):
        return len(self.data)

    def _get_random_index(self):
        return random.randrange(0, len(self.data))

    def __getitem__(self, index):
        current_index = index
        while True:
            row = self.data[current_index]
            duration = row["duration"]
            if isinstance(duration, str):
                duration = float(duration)

            # Filter by duration / DNSMOS; on rejection, retry a random index.
            if self.filter_config is not None and not self.filter_config.disabled:
                if duration < self.filter_config.min_seq_len:
                    current_index = self._get_random_index()
                    continue
                if duration > self.filter_config.max_seq_len:
                    current_index = self._get_random_index()
                    continue
                if "dnsmos" in row and row["dnsmos"] < self.filter_config.min_dnsmos:
                    current_index = self._get_random_index()
                    continue

            try:
                path = row["audio_path"]
                if "LibriTTS" not in path:
                    # Rebuild the Emilia hierarchical path from the clip filename,
                    # e.g. ZH_B00046_S00437_W000001.mp3 ->
                    #   <ROOT>/ZH/ZH_B00046/ZH_B00046_S00437/mp3/<filename>
                    filename = os.path.basename(path)
                    stem = os.path.splitext(filename)[0]
                    parts = stem.split("_")
                    lang = parts[0]                  # ZH
                    book = "_".join(parts[:2])       # ZH_B00046
                    speaker = "_".join(parts[:3])    # ZH_B00046_S00437
                    path = os.path.join(
                        EMILIA_DATASET_ROOT, lang, book, speaker, "mp3", filename
                    )
                signal, sample_rate = torchaudio.load(path)
                signal = signal.unsqueeze(0)  # (1, 1, T)
                if sample_rate != self.target_sample_rate:
                    signal = torchaudio.functional.resample(
                        signal, orig_freq=sample_rate, new_freq=self.target_sample_rate
                    )
                if self.pad_multiple_of is not None:
                    remainder = signal.shape[-1] % self.pad_multiple_of
                    pad_len = self.pad_multiple_of - remainder if remainder != 0 else 0
                    signal = F.pad(signal, (0, pad_len))
                ldm_feature = signal.squeeze(0)  # (1, T)
            except Exception as e:
                print(e)
                current_index = self._get_random_index()
                continue

            break

        # ORW: for RL datasets (with a gen_text column), text is the prompt text
        # followed by the target text; the raw gen_text is passed through to
        # collate_fn as the ASR reference used to compute the rollout reward.
        if "gen_text" in row:
            text = row["text"] + " " + row["gen_text"]
        else:
            text = row["text"]

        # Language: prefer the dataset's own `language` column; fall back to
        # inferring zh/en from the audio path (e.g. Emilia_old only carries
        # audio_path/text/duration).
        if "language" in row:
            lang = row["language"]
        else:
            lang = detect_lang_from_path(row.get("audio_path") or row.get("path"))

        return {
            "mel_spec": ldm_feature,
            "text": text,
            "gen_text": row["gen_text"] if "gen_text" in row else None,
            "lang": lang,
        }

    def collate_fn(self, batch):
        mel_specs = [item["mel_spec"] for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()
        mel_mask = ~make_pad_mask(mel_lengths, max_len=max_mel_length)

        # Right-pad each waveform with zeros to the batch max length.
        padded_mel_specs = []
        for spec in mel_specs:
            padding = (0, max_mel_length - spec.size(-1))
            padded_mel_specs.append(F.pad(spec, padding, value=0))
        mel_specs = torch.stack(padded_mel_specs)

        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])
        max_text_length = text_lengths.amax()
        text_mask = ~make_pad_mask(text_lengths, max_text_length)

        # ORW: pass gen_text (target synthesis text) through so DiTAR_ORW can
        # compute the ASR-based reward after rollout.
        gen_text = None
        if "gen_text" in batch[0]:
            gen_text = [item["gen_text"] for item in batch]

        # ORW: pass lang (zh/en inferred from the audio path) through so the
        # reward side can compute per-language WER.
        lang = None
        if "lang" in batch[0]:
            lang = [item["lang"] for item in batch]

        return dict(
            mel=mel_specs,
            mel_lengths=mel_lengths,
            mel_mask=mel_mask,
            text=text,
            text_lengths=text_lengths,
            text_mask=text_mask,
            gen_text=gen_text,
            lang=lang,
        )


class CustomLDMEvalDataset(Dataset):
    """Evaluation dataset (batch_size=1 inference). Loads prompt / ground-truth
    waveforms for the LibriSpeech-PC and seed-tts-eval test sets."""

    def __init__(
        self,
        meta_info,
        prompt_wav_path,
        durations=None,
        testset_name=None,
        audio_type: str = "vae_online",
        target_sample_rate: int = 16000,
        pad_multiple_of: int = None,
    ):
        self.meta_info = meta_info
        self.prompt_wav_path = prompt_wav_path
        self.durations = durations
        self.testset_name = testset_name
        self.audio_type = audio_type
        self.target_sample_rate = target_sample_rate
        self.pad_multiple_of = pad_multiple_of

    def _get_ls_pc_feature_path(self, wav_path):
        file_name_no_ext = os.path.splitext(os.path.basename(wav_path))[0]
        if "-" in file_name_no_ext:
            spk, chapter, _ = file_name_no_ext.split("-")
        elif "_" in file_name_no_ext:
            spk, chapter, _, _ = file_name_no_ext.split("_")
        return os.path.join(self.prompt_wav_path, spk, chapter, f"{file_name_no_ext}.flac")

    def _load_feature(self, feature_path):
        signal, sample_rate = torchaudio.load(feature_path)
        signal = signal.unsqueeze(0)  # (1, 1, T)
        if sample_rate != self.target_sample_rate:
            signal = torchaudio.functional.resample(
                signal, orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
        if self.pad_multiple_of is not None:
            remainder = signal.shape[-1] % self.pad_multiple_of
            pad_len = self.pad_multiple_of - remainder if remainder != 0 else 0
            if pad_len > 0:
                signal = F.pad(signal, (0, pad_len))
        return signal

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, index):
        item = self.meta_info[index]
        if "seedtts" in self.testset_name:
            key, prompt_text, prompt_wav, gt_text, gt_wav = item
            prompt_phn = prompt_text
            gt_phn = gt_text
            use_seedtts_paths = True
        elif self.testset_name == "ls_pc_test_clean" or self.testset_name == "train-set":
            key, prompt_text, prompt_wav, gt_text, gt_wav = item
            prompt_phn = prompt_text
            gt_phn = gt_text
            use_seedtts_paths = False
        else:
            raise NotImplementedError(f"Unknown testset: {self.testset_name!r}")

        if use_seedtts_paths:
            feature_path = prompt_wav
        else:
            feature_path = self._get_ls_pc_feature_path(prompt_wav)

        ldm_feature = self._load_feature(feature_path).squeeze(0)  # (1, T)

        return {
            "key": key,
            "prompt_phn": prompt_phn + gt_phn,
            "prompt_vae": ldm_feature,
            "gt_text": gt_text,
        }

    def collate_fn(self, batch):  # batch_size=1
        key = [item["key"] for item in batch]

        prompt_vae = torch.stack([item["prompt_vae"] for item in batch])

        vae_lengths = torch.LongTensor([spec.shape[-1] for spec in prompt_vae])
        max_vae_length = vae_lengths.amax()
        vae_mask = ~make_pad_mask(vae_lengths, max_len=max_vae_length)

        phn = [item["prompt_phn"] for item in batch]
        phn_lengths = torch.LongTensor([len(item) for item in phn])
        max_phn_length = phn_lengths.amax()
        phn_mask = ~make_pad_mask(phn_lengths, max_phn_length).flip(dims=(1,))

        gt_text = [item["gt_text"] for item in batch]

        return dict(
            key=key,
            prompt_vae=prompt_vae,
            vae_lengths=vae_lengths,
            vae_mask=vae_mask,
            phn=phn,
            phn_lengths=phn_lengths,
            phn_mask=phn_mask,
            gt_text=gt_text,
        )


def load_ldm_dataset(
    dataset_path: str,
    audio_type: str,
    filter_config: LDMDataFilterConfig = None,
    target_sample_rate: int = 16000,
):
    train_dataset = load_from_disk(f"{dataset_path}")
    try:
        with open(f"{dataset_path}/duration.json", "r", encoding="utf-8") as f:
            durations = json.load(f)["duration"]
    except Exception:
        durations = train_dataset["duration"]

    return CustomLDMDataset(
        train_dataset,
        filter_config=filter_config,
        durations=durations,
        audio_type=audio_type,
        target_sample_rate=target_sample_rate,
    )

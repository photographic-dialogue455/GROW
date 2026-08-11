from __future__ import annotations

import os
import sys
from importlib.resources import files

import torch

import jieba
from pypinyin import lazy_pinyin, Style
from transformers import AutoTokenizer


# helpers
def is_debugging():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d

# tensor helpers
def get_mask_from_lengths(lengths, max_len=None, r=1):  # like make_pad_mask but with True/False inverted
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % r != 0:
        max_len = max_len + r - max_len % r
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

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
    >>> lengths = torch.tensor([0, 2, 1, 4])
    >>> make_pad_mask(lengths)
    tensor([[True,  True,  True,  True,  True],
            [False, False, True,  True,  True],
            [False, True,  True,  True,  True],
            [False, False, False, False, True]])
    """

    if type(lengths) is list:
        lengths = torch.tensor(lengths)
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


# char tokenizer, based on custom dataset's extracted .txt file


# Get tokenizer


def get_tokenizer(dataset_name=None, tokenizer: str = "pinyin", tokenizer_path=None, model_id_or_path=None):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char", "g2p"]:
        if tokenizer_path is None:
            tokenizer_path = os.path.join(files("DiTAR").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map) + 1
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    elif tokenizer == "bpe":
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id_or_path)
        tokenizer_path = os.path.join(files("DiTAR").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        vocab_char_map = {}
        vocab_size = len(tokenizer)
    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

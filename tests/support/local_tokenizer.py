"""A Hugging Face tokenizer saved to a directory, so a load from a path runs offline."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


def save_tokenizer(directory: Path) -> Path:
    """Save a two-word tokenizer to ``directory`` and return it."""
    vocab = {"[UNK]": 0, "hi": 1}
    model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=model).save_pretrained(directory)
    return directory

"""Renders share one tokenizer across threads: its encodes must only read the backend.

transformers sets truncation and padding on the shared Rust backend for the length of
an encode that asks for them, and resets them on the next one that finds them set. An
encode that asks for neither, with nothing set, writes nothing, so threads can share
the tokenizer. That is how the transformers versions pyproject.toml allows behave; this
test is what qualifies a new one before the bound is widened.
"""

from support.recording_tokenizer import record_writes
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "the": 1, "court": 2, "held": 3, "<|im_end|>": 4}
    model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    model.pre_tokenizer = Whitespace()
    loaded = PreTrainedTokenizerFast(tokenizer_object=model)
    loaded.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
    return loaded


def test_the_calls_renderers_make_write_nothing_to_the_backend():
    tok = tokenizer()
    tok("the court", add_special_tokens=False)  # the first call settles any initial state
    writes = record_writes(tok)
    for _ in range(3):
        encoded = tok("the court held", add_special_tokens=False, return_offsets_mapping=True)
        assert encoded["input_ids"] == [1, 2, 3] and len(encoded["offset_mapping"]) == 3
        assert tok.encode("held the court<|im_end|>", add_special_tokens=False) == [3, 1, 2, 4]
        assert tok.decode([1, 2, 4], skip_special_tokens=False)
        assert tok.convert_tokens_to_ids("<|im_end|>") == 4
    assert writes == []


def test_an_encode_that_asks_for_truncation_writes_and_the_next_one_resets_it():
    """What sharing must avoid, and what proves the recording sees it."""
    tok = tokenizer()
    writes = record_writes(tok)
    assert tok("the court held", truncation=True, max_length=2)["input_ids"] == [1, 2]
    assert tok("the court held", add_special_tokens=False)["input_ids"] == [1, 2, 3]
    assert writes == ["enable_truncation", "no_truncation"]

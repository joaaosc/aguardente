"""Testes de carregamento de textos e preparação de lotes de calibração."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aguardente.calibration import (
    ensure_pad_token,
    load_texts_from_file,
    make_batches,
)
from aguardente.errors import AguardenteError


def test_ensure_pad_token_sets_eos_when_none():
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    res = ensure_pad_token(tokenizer)
    assert res.pad_token == "<eos>"


def test_ensure_pad_token_keeps_existing():
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    res = ensure_pad_token(tokenizer)
    assert res.pad_token == "<pad>"


def test_load_texts_from_file_with_paragraphs(tmp_path):
    p = tmp_path / "data.txt"
    par1 = "A" * 250
    par2 = "B" * 300
    short = "C" * 50
    p.write_text(f"{par1}\n\n{short}\n\n{par2}", encoding="utf-8")

    texts = load_texts_from_file(str(p), min_chars=200)
    assert texts == [par1, par2]


def test_load_texts_from_file_with_line_fallback(tmp_path):
    p = tmp_path / "data_lines.txt"
    line1 = "Primeira linha com texto suficiente para compor o bloco de calibração. "
    line2 = "Segunda linha dando continuidade ao conteúdo do arquivo de texto comum. "
    line3 = "Terceira linha finalizando o conjunto com mais de duzentos caracteres no total."
    p.write_text(f"{line1}\n{line2}\n{line3}", encoding="utf-8")

    texts = load_texts_from_file(str(p), min_chars=150)
    assert len(texts) >= 1
    assert len(texts[0]) >= 150


def test_load_texts_from_file_missing_raises():
    with pytest.raises(AguardenteError, match="não encontrado"):
        load_texts_from_file("/caminho/inexistente/data.txt")


def test_load_texts_from_file_too_short_raises(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("Muito curto", encoding="utf-8")
    with pytest.raises(AguardenteError, match="não contém parágrafos"):
        load_texts_from_file(str(p), min_chars=200)


def test_make_batches_creates_fixed_batches_and_masks():
    torch = pytest.importorskip("torch")

    class DummyTokenizer:
        def __init__(self):
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"

        def __call__(self, chunk, return_tensors, padding, truncation, max_length):
            bs = len(chunk)
            return {
                "input_ids": torch.randint(0, 100, (bs, max_length)),
                "attention_mask": torch.ones((bs, max_length), dtype=torch.int8),
            }

    texts = ["Texto 1", "Texto 2", "Texto 3", "Texto 4", "Texto 5 (descartado)"]
    batches = list(make_batches(DummyTokenizer(), texts, batch_size=2, seq_len=16))

    assert len(batches) == 2
    for b in batches:
        assert b["input_ids"].shape == (2, 16)
        assert b["attention_mask"].shape == (2, 16)

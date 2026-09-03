"""Testes para o cálculo de métricas de qualidade e perplexidade."""

import pytest

from aguardente.verify import Perplexity, perplexity, recovery_fraction


def test_perplexity_dataclass_str():
    p = Perplexity(value=14.25, tokens=1000, windows=2)
    assert "14.25" in str(p)
    assert "1,000 tokens" in str(p)


def test_perplexity_rejects_too_short_text():
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.return_value.input_ids = torch.tensor([[1]])
    model = MagicMock()

    with pytest.raises(ValueError, match="texto curto demais"):
        perplexity(model, tokenizer, "a", device="cpu")


def test_perplexity_sliding_window_counts_exact_tokens():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).eval()

    class DummyTokenizer:
        def __call__(self, text, return_tensors):
            # Gera 50 tokens
            return type("Enc", (), {"input_ids": torch.randint(0, 64, (1, 50))})()

    res = perplexity(model, DummyTokenizer(), "texto longo de teste", max_length=20, stride=10, device="cpu")
    assert res.value > 0.0
    assert res.windows > 1
    assert res.tokens == 49  # 50 tokens = 49 previsões de próximo token


@pytest.mark.parametrize("teacher,pruned,recovered,expected", [
    (10.0, 20.0, 15.0, 0.5),
    (10.0, 20.0, 10.0, 1.0),
    (10.0, 20.0, 20.0, 0.0),
])
def test_recovery_fraction_basic(teacher, pruned, recovered, expected):
    assert recovery_fraction(teacher, pruned, recovered) == pytest.approx(expected)

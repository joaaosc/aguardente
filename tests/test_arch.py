"""A contagem de parâmetros precisa bater com o que o Hugging Face reporta.

Os configs abaixo são cópias literais dos campos relevantes de modelos reais,
com o total vindo de `safetensors.total` da API. Se a fórmula divergir mais de
1%, o planejamento de poda deixa de ser confiável.
"""

import pytest

from aguardente.arch import Arch, count_params, reconcile

# (nome, config parcial, safetensors.total real)
REAL_MODELS = [
    (
        "Qwen/Qwen3-0.6B",
        dict(model_type="qwen3", hidden_size=1024, intermediate_size=3072,
             num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
             head_dim=128, vocab_size=151936, tie_word_embeddings=True),
        751_632_384,
        True,   # grava lm_head apesar de tie_word_embeddings=True
    ),
    (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        dict(model_type="llama", hidden_size=576, intermediate_size=1536,
             num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
             vocab_size=49152, tie_word_embeddings=True),
        134_515_008,
        False,
    ),
]


@pytest.mark.parametrize("name,cfg,reported,lm_head", REAL_MODELS)
def test_count_matches_hf_total(name, cfg, reported, lm_head):
    """A formula tem de bater ao parametro com o safetensors.total real."""
    arch = Arch.from_hf_config(cfg)
    err = reconcile(arch, reported, lm_head_materialized=lm_head)
    assert err == 0.0, f"{name}: erro {err:+.4%} (contou {count_params(arch).total:,})"


def test_stored_vs_logical_diverge_when_lm_head_materialized():
    """Qwen3-0.6B grava lm_head mesmo com tie_word_embeddings — o config nao preve isso."""
    from aguardente.arch import count_stored
    arch = Arch.from_hf_config(REAL_MODELS[0][1])
    logical = count_params(arch).total
    stored = count_stored(arch, lm_head_materialized=True)
    assert stored - logical == arch.vocab_size * arch.hidden_size
    assert stored == 751_632_384


def test_mlp_dominates_in_qwen3_4b():
    """A ordem de poda depende disto: o MLP tem de ser a maior fatia."""
    arch = Arch.from_hf_config(dict(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=True,
    ))
    shares = count_params(arch).shares()
    assert shares["mlp"] > shares["attention"] > shares["embeddings"]
    assert shares["mlp"] > 0.6


def test_gqa_must_divide():
    from aguardente.errors import UnsupportedArchitecture
    with pytest.raises(UnsupportedArchitecture):
        Arch(hidden_size=512, intermediate_size=1024, num_hidden_layers=4,
             num_attention_heads=10, num_key_value_heads=3, head_dim=64,
             vocab_size=1000)


def test_untied_embeddings_counted_twice():
    base = dict(hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                num_attention_heads=8, num_key_value_heads=8, head_dim=64,
                vocab_size=10_000)
    tied = count_params(Arch(**base, tie_word_embeddings=True)).total
    untied = count_params(Arch(**base, tie_word_embeddings=False)).total
    assert untied - tied == 10_000 * 512

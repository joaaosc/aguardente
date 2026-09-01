"""Testes das guardas de arquitetura em Arch.from_hf_config."""

import pytest

from aguardente.arch import Arch
from aguardente.errors import UnsupportedArchitecture

DENSO = dict(model_type="qwen3", hidden_size=1024, intermediate_size=3072,
             num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
             head_dim=128, vocab_size=151936, tie_word_embeddings=True)


def test_config_denso_continua_aceito():
    a = Arch.from_hf_config(DENSO)
    assert a.hidden_size == 1024 and a.num_hidden_layers == 28


@pytest.mark.parametrize("chave", ["num_experts", "n_routed_experts", "num_local_experts",
                                   "moe_intermediate_size", "num_experts_per_tok",
                                   "n_shared_experts"])
def test_moe_rejeitado_com_mensagem(chave):
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {chave: 256})
    assert "MoE" in e.value.message and chave in e.value.message
    assert "densa" in e.value.hint


@pytest.mark.parametrize("chave", ["vision_config", "vision_n_layers", "vision_tower"])
def test_multimodal_rejeitado_com_mensagem(chave):
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {chave: 32})
    assert "multimodal" in e.value.message and chave in e.value.message


def test_visual_como_subconfig_e_rejeitado():
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {"visual": {"depth": 32}})
    assert "multimodal" in e.value.message and "visual" in e.value.message


def test_visual_escalar_nao_dispara_guarda():
    """`visual` é genérica: só um dicionário aninhado indica torre de visão."""
    assert Arch.from_hf_config(DENSO | {"visual": 1}).hidden_size == 1024


def test_chave_moe_zerada_nao_dispara_guarda():
    """Um zero explícito é ausência, não presença de especialistas."""
    assert Arch.from_hf_config(DENSO | {"num_experts": 0}).hidden_size == 1024


@pytest.mark.parametrize("cfg_extra, marca", [
    ({"quantization_config": {"quant_method": "gptq", "bits": 4}}, "gptq"),
    ({"quantization_config": {"bits": 4}}, "quantization_config"),
    ({"dtype": "float8_e4m3fn"}, "float8"),
    ({"torch_dtype": "int8"}, "int8"),
])
def test_pesos_quantizados_rejeitados(cfg_extra, marca):
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | cfg_extra)
    assert "quantizados" in e.value.message and marca in e.value.message


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_dtype_denso_continua_aceito(dtype):
    assert Arch.from_hf_config(DENSO | {"dtype": dtype}).hidden_size == 1024


def test_head_dim_nao_exato_sem_declaracao_e_recusado():
    """Sem head_dim explícito, hidden/heads truncava e falseava a contagem."""
    cfg = {k: v for k, v in DENSO.items() if k != "head_dim"} | {"hidden_size": 1000}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert "head_dim" in e.value.message


def test_head_dim_declarado_dispensa_divisao_exata():
    cfg = DENSO | {"hidden_size": 1000, "head_dim": 64}
    assert Arch.from_hf_config(cfg).head_dim == 64


def test_tie_word_embeddings_segue_padrao_do_transformers():
    """Config que omite a chave compartilha embeddings, como no transformers."""
    cfg = {k: v for k, v in DENSO.items() if k != "tie_word_embeddings"}
    assert Arch.from_hf_config(cfg).tie_word_embeddings is True


@pytest.mark.parametrize("chave", ["use_qk_norm", "qk_norm", "attention_qk_norm",
                                   "qk_layernorm"])
def test_qk_norm_explicito_tem_precedencia_sobre_a_heuristica(chave):
    cfg = DENSO | {"model_type": "llama", chave: True}
    assert Arch.from_hf_config(cfg).qk_norm is True
    assert Arch.from_hf_config(DENSO | {chave: False}).qk_norm is False


def test_moe_tem_precedencia_sobre_multimodal():
    """Um config MoE multimodal deve reportar primeiro a razão que invalida a contagem."""
    cfg = DENSO | {"n_routed_experts": 256, "vision_config": {}}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert "MoE" in e.value.message


def test_chave_moe_nula_nao_dispara_guarda():
    """Configs que declaram a chave como null não são MoE."""
    assert Arch.from_hf_config(DENSO | {"num_experts": None}).hidden_size == 1024


def test_config_aninhado_rejeitado():
    cfg = {"model_type": "multi_modality", "language_config": {"hidden_size": 2048},
           "aligner_config": {}}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert "language_config" in e.value.message


@pytest.mark.parametrize("faltante", ["hidden_size", "num_attention_heads", "intermediate_size",
                                      "num_hidden_layers", "vocab_size"])
def test_dimensao_ausente_gera_erro_amigavel(faltante):
    """Regressão: intermediate_size ausente produzia KeyError cru em vez de mensagem."""
    cfg = {k: v for k, v in DENSO.items() if k != faltante}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert faltante in e.value.message
    assert e.value.hint


def test_dimensao_invalida_gera_erro_amigavel():
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {"num_hidden_layers": "vinte e oito"})
    assert "inválido" in e.value.message

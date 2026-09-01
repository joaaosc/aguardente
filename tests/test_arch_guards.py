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


@pytest.mark.parametrize("chave", ["vision_config", "vision_n_layers", "vision_tower",
                                   "visual"])
def test_torre_de_visao_nao_impede_mais_a_leitura(chave):
    """A torre é descartada pela extração; sua presença não invalida as dimensões."""
    assert Arch.from_hf_config(DENSO | {chave: {"depth": 32}}).hidden_size == 1024


def test_dimensoes_aninhadas_sao_alcancadas():
    """Num VLM as dimensões do decoder ficam sob text_config ou equivalente."""
    a = Arch.from_hf_config({
        "model_type": "idefics3",
        "vision_config": {"depth": 27},
        "text_config": DENSO,
    })
    assert a.hidden_size == 1024 and a.num_hidden_layers == 28


@pytest.mark.parametrize("chave", ["text_config", "language_config", "llm_config"])
def test_sub_config_e_encontrado_por_qualquer_das_chaves(chave):
    assert Arch.from_hf_config({"model_type": "x", chave: DENSO}).hidden_size == 1024


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


def test_moe_multimodal_reporta_o_moe():
    """A torre seria descartada; o MoE é o que de fato invalida a contagem."""
    cfg = DENSO | {"n_routed_experts": 256, "vision_config": {}}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert "MoE" in e.value.message


def test_chave_moe_nula_nao_dispara_guarda():
    """Configs que declaram a chave como null não são MoE."""
    assert Arch.from_hf_config(DENSO | {"num_experts": None}).hidden_size == 1024


def test_moe_no_sub_config_e_recusado_dizendo_onde():
    """Um sinal de MoE no language_config conta tanto quanto um no topo."""
    cfg = {"model_type": "deepseek_vl_v2",
           "language_config": DENSO | {"n_routed_experts": 72}}
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(cfg)
    assert "MoE" in e.value.message and "language_config" in e.value.message


@pytest.mark.parametrize("chave", ["kv_lora_rank", "q_lora_rank"])
def test_atencao_latente_e_recusada(chave):
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {chave: 512})
    assert "MLA" in e.value.message and chave in e.value.message


@pytest.mark.parametrize("chave", ["cross_attention_layers", "cross_attn_layers"])
def test_cross_attention_intercalada_e_recusada(chave):
    """O mllama intercala blocos de tipos diferentes; a seleção de camadas quebraria."""
    with pytest.raises(UnsupportedArchitecture) as e:
        Arch.from_hf_config(DENSO | {chave: [3, 8, 13]})
    assert "cross-attention" in e.value.message


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

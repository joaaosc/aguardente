"""Reconstrução das dimensões a partir das formas dos tensores.

As formas usadas reproduzem as dos checkpoints reais: DeepSeek-VL-7B (MHA, sem
bias, lm_head materializado) e Qwen2.5-VL-3B (GQA, com bias em q/k/v,
embeddings amarradas).
"""

import pytest

from aguardente.arch import count_params
from aguardente.errors import UnsupportedArchitecture
from aguardente.textonly import (arch_from_shapes, discover_layout, reconcile,
                                 text_config)


def t(*shape):
    return {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}


def checkpoint(prefixo, *, hidden, inter, camadas, vocab, q_out, kv_out,
               bias=False, lm_head=False, qk_norm=False):
    h = {f"{prefixo}embed_tokens.weight": t(vocab, hidden),
         f"{prefixo}norm.weight": t(hidden)}
    for i in range(camadas):
        base = f"{prefixo}layers.{i}."
        h[base + "self_attn.q_proj.weight"] = t(q_out, hidden)
        h[base + "self_attn.k_proj.weight"] = t(kv_out, hidden)
        h[base + "self_attn.v_proj.weight"] = t(kv_out, hidden)
        h[base + "self_attn.o_proj.weight"] = t(hidden, q_out)
        h[base + "mlp.gate_proj.weight"] = t(inter, hidden)
        h[base + "mlp.up_proj.weight"] = t(inter, hidden)
        h[base + "mlp.down_proj.weight"] = t(hidden, inter)
        h[base + "input_layernorm.weight"] = t(hidden)
        h[base + "post_attention_layernorm.weight"] = t(hidden)
        if bias:
            h[base + "self_attn.q_proj.bias"] = t(q_out)
            h[base + "self_attn.k_proj.bias"] = t(kv_out)
            h[base + "self_attn.v_proj.bias"] = t(kv_out)
        if qk_norm:
            h[base + "self_attn.q_norm.weight"] = t(hidden // (q_out // kv_out or 1))
            h[base + "self_attn.k_norm.weight"] = t(hidden // (q_out // kv_out or 1))
    if lm_head:
        h[prefixo.removesuffix("model.") + "lm_head.weight"] = t(vocab, hidden)
    return h


def deepseek(camadas=30):
    """Decoder do DeepSeek-VL-7B: MHA pura, sem bias, lm_head materializado."""
    return checkpoint("language_model.model.", hidden=4096, inter=11008,
                      camadas=camadas, vocab=102400, q_out=4096, kv_out=4096,
                      lm_head=True)


def qwen(camadas=36):
    """Decoder do Qwen2.5-VL-3B: GQA 16/2, bias em q/k/v, embeddings amarradas."""
    return checkpoint("model.", hidden=2048, inter=11008, camadas=camadas,
                      vocab=151936, q_out=2048, kv_out=256, bias=True)


def montar(h, cfg=None):
    headers = {"m.safetensors": h}
    return arch_from_shapes(headers, discover_layout(headers), cfg=cfg)


# ------------------------------------------------------- dimensões das formas


def test_dimensoes_do_deepseek_sem_config_completo():
    """O language_config do DeepSeek-VL declara só camadas e vocabulário."""
    a = montar(deepseek(), cfg={"model_type": "llama", "num_hidden_layers": 30,
                                "vocab_size": 102400})

    assert (a.hidden_size, a.intermediate_size, a.num_hidden_layers) == (4096, 11008, 30)
    assert (a.num_attention_heads, a.num_key_value_heads, a.head_dim) == (32, 32, 128)
    assert a.vocab_size == 102400
    assert a.tie_word_embeddings is False
    assert a.attention_bias is False


def test_dimensoes_do_qwen_com_gqa_e_bias():
    a = montar(qwen(), cfg={"model_type": "qwen2_5_vl", "head_dim": 128})

    assert (a.hidden_size, a.intermediate_size, a.num_hidden_layers) == (2048, 11008, 36)
    assert (a.num_attention_heads, a.num_key_value_heads) == (16, 2)
    assert a.tie_word_embeddings is True
    assert a.attention_bias is True


def test_lm_head_materializado_nao_desfaz_compartilhamento_do_config():
    h = checkpoint("model.", hidden=32, inter=64, camadas=2, vocab=128,
                   q_out=32, kv_out=16, lm_head=True)
    headers = {"m.safetensors": h}
    layout = discover_layout(headers)
    a = arch_from_shapes(headers, layout,
                         cfg={"head_dim": 8, "tie_word_embeddings": True})
    assert a.tie_word_embeddings is True
    assert count_params(a).lm_head == 0
    assert layout.kept_params - count_params(a).total == 128 * 32
    assert reconcile(a, layout) == 0


def test_cabeca_independente_ausente_nao_e_inventada():
    with pytest.raises(UnsupportedArchitecture, match="lm_head está ausente"):
        montar(qwen(2), cfg={"head_dim": 128, "tie_word_embeddings": False})


def test_qk_norm_vem_das_formas_nao_do_model_type():
    a = montar(checkpoint("model.", hidden=1024, inter=3072, camadas=4, vocab=1000,
                          q_out=1024, kv_out=512, qk_norm=True),
               cfg={"head_dim": 128})

    assert a.qk_norm is True


# --------------------------------------------------------- reconciliação


def test_reconciliacao_do_deepseek_e_exata():
    h = deepseek()
    layout = discover_layout({"m.safetensors": h})
    a = arch_from_shapes({"m.safetensors": h}, layout, cfg={"model_type": "llama"})

    assert layout.kept_params == 6_910_365_696
    assert reconcile(a, layout) == 0


def test_reconciliacao_do_qwen_e_exata_com_bias():
    """Sem o termo de bias a contagem erraria por 92.160 — (2048 + 2×256) × 36."""
    h = qwen()
    layout = discover_layout({"m.safetensors": h})
    a = arch_from_shapes({"m.safetensors": h}, layout, cfg={"head_dim": 128})

    assert layout.kept_params == 3_085_938_688
    assert reconcile(a, layout) == 0
    assert count_params(a.with_(attention_bias=False)).total == 3_085_846_528


# ------------------------------------------------ resolução do número de cabeças


def test_head_dim_do_config_tem_precedencia():
    a = montar(qwen(4), cfg={"head_dim": 64, "num_attention_heads": 999})
    assert a.head_dim == 64 and a.num_attention_heads == 32


def test_num_attention_heads_usado_quando_nao_ha_head_dim():
    a = montar(deepseek(2), cfg={"num_attention_heads": 64})
    assert a.head_dim == 64 and a.num_attention_heads == 64


def test_default_de_classe_e_conferido_contra_as_formas():
    """32 cabeças de LlamaConfig fecham com q_out 4096: head_dim 128."""
    a = montar(deepseek(2), cfg={"model_type": "llama"})
    assert a.head_dim == 128 and a.num_attention_heads == 32


def test_sem_head_dim_sem_cabecas_e_sem_default_conhecido():
    with pytest.raises(UnsupportedArchitecture) as e:
        montar(deepseek(2), cfg={"model_type": "arquitetura_nova"})

    assert "arquitetura_nova" in e.value.message
    assert "head_dim" in e.value.message


def test_cabecas_do_config_incompativeis_com_q_proj():
    with pytest.raises(UnsupportedArchitecture) as e:
        montar(deepseek(2), cfg={"num_attention_heads": 30})

    assert "4096" in e.value.message and "30" in e.value.message


def test_head_dim_que_nao_divide_as_projecoes():
    with pytest.raises(UnsupportedArchitecture) as e:
        montar(qwen(2), cfg={"head_dim": 100})

    assert "não divide" in e.value.message and "2048" in e.value.message


def test_gqa_com_grupos_nao_divisores():
    h = checkpoint("model.", hidden=1024, inter=2048, camadas=2, vocab=1000,
                   q_out=1024, kv_out=384)
    with pytest.raises(UnsupportedArchitecture) as e:
        montar(h, cfg={"head_dim": 128})

    assert "8 cabeças" in e.value.message and "3 grupos" in e.value.message


# ----------------------------------------------------------- text_config


def test_config_de_topo_e_o_proprio():
    cfg, chave = text_config({"hidden_size": 2048, "model_type": "qwen2_5_vl",
                              "vision_config": {"depth": 32}})
    assert chave == "" and cfg["hidden_size"] == 2048


@pytest.mark.parametrize("chave", ["text_config", "language_config", "llm_config"])
def test_config_aninhado_e_encontrado(chave):
    cfg, achada = text_config({"model_type": "multi_modality",
                               chave: {"hidden_size": 4096, "model_type": "llama"}})
    assert achada == chave and cfg["hidden_size"] == 4096


def test_campos_do_topo_sao_herdados_quando_faltam():
    cfg, _ = text_config({"model_type": "idefics3", "tie_word_embeddings": False,
                          "eos_token_id": 2,
                          "text_config": {"hidden_size": 2048, "vocab_size": 49280}})

    assert cfg["tie_word_embeddings"] is False and cfg["eos_token_id"] == 2
    assert cfg["vocab_size"] == 49280


def test_sub_config_vence_o_topo_em_conflito():
    cfg, _ = text_config({"vocab_size": 1, "text_config": {"hidden_size": 8,
                                                           "vocab_size": 999}})
    assert cfg["vocab_size"] == 999


def test_dois_sub_configs_e_ambiguidade():
    with pytest.raises(UnsupportedArchitecture) as e:
        text_config({"text_config": {"hidden_size": 8},
                     "language_config": {"hidden_size": 16}})

    assert "text_config" in e.value.message and "language_config" in e.value.message


def test_sem_dimensoes_em_lugar_nenhum():
    with pytest.raises(UnsupportedArchitecture) as e:
        text_config({"model_type": "algo", "vision_config": {"depth": 3}})

    assert "text_config" in e.value.message


# ------------------------------------------------------- normas por camada


def gemma(camadas=34):
    """Gemma 3 normaliza também a saída de cada bloco: quatro normas por camada."""
    h = checkpoint("language_model.model.", hidden=2560, inter=10240,
                   camadas=camadas, vocab=262208, q_out=2048, kv_out=1024)
    for i in range(camadas):
        base = f"language_model.model.layers.{i}."
        h[base + "pre_feedforward_layernorm.weight"] = t(2560)
        h[base + "post_feedforward_layernorm.weight"] = t(2560)
        h[base + "self_attn.q_norm.weight"] = t(256)
        h[base + "self_attn.k_norm.weight"] = t(256)
    return h


def test_normas_extras_do_gemma_sao_contadas():
    """Sem isto a contagem erraria 174.080 no Gemma 3 4B — 2 × 2560 × 34."""
    h = gemma()
    layout = discover_layout({"m.safetensors": h})
    a = arch_from_shapes({"m.safetensors": h}, layout, cfg={"head_dim": 256})

    assert a.norms_per_layer == 4
    assert a.qk_norm is True
    assert reconcile(a, layout) == 0


def test_bias_de_q_proj_nao_e_confundido_com_norma():
    """No Qwen2 e no InternVL3, q_proj.bias tem exatamente o tamanho de hidden."""
    h = qwen(4)
    layout = discover_layout({"m.safetensors": h})
    a = arch_from_shapes({"m.safetensors": h}, layout, cfg={"head_dim": 128})

    assert a.norms_per_layer == 2
    assert a.attention_bias is True
    assert reconcile(a, layout) == 0


def test_norms_per_layer_do_config_segue_a_familia():
    """`from_hf_config` não tem as formas: usa o model_type para a família Gemma."""
    from aguardente.arch import Arch

    base = dict(hidden_size=2560, intermediate_size=10240, num_hidden_layers=34,
                num_attention_heads=8, num_key_value_heads=4, head_dim=256,
                vocab_size=262208)
    assert Arch.from_hf_config(base | {"model_type": "gemma3_text"}).norms_per_layer == 4
    assert Arch.from_hf_config(base | {"model_type": "llama"}).norms_per_layer == 2

"""Descoberta do decoder causal dentro de um checkpoint multimodal.

Os cabeçalhos sintéticos reproduzem a nomenclatura real dos repositórios
citados, lida dos respectivos `model.safetensors.index.json`.
"""

import pytest

from aguardente.errors import UnsupportedArchitecture
from aguardente.textonly import SIGNATURE, discover_layout


def t(*shape):
    return {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}


def decoder(prefixo, camadas, *, sufixos=None, hidden=64, inter=128, vocab=1000):
    """Tensores de um decoder causal denso sob o prefixo indicado."""
    h = {f"{prefixo}embed_tokens.weight": t(vocab, hidden),
         f"{prefixo}norm.weight": t(hidden)}
    for i in range(camadas):
        for sufixo in (sufixos or SIGNATURE):
            h[f"{prefixo}layers.{i}.{sufixo}"] = t(hidden, hidden)
    return h


def visao_deepseek():
    """Torre híbrida e projetor do DeepSeek-VL: `aligner.layers.N` é o falso positivo."""
    h = {f"aligner.layers.{i}.{p}": t(64) for i in range(2) for p in ("weight", "bias")}
    h["aligner.high_up_proj.weight"] = t(64, 64)
    for i in range(3):
        for p in ("attn.qkv.weight", "attn.proj.weight", "norm1.weight"):
            h[f"vision_model.vision_tower_high.vision_tower.blocks.{i}.{p}"] = t(64, 64)
    return h


def visao_qwen():
    """Torre do Qwen2.5-VL: `visual.blocks.N`, que nem casa `layers.N.`."""
    return {f"visual.blocks.{i}.{p}": t(64, 64)
            for i in range(3)
            for p in ("attn.qkv.weight", "mlp.down_proj.weight", "norm1.weight")}


# ------------------------------------------------------------ prefixos reais


def test_prefixo_deepseek():
    h = decoder("language_model.model.", 30) | visao_deepseek()
    h["language_model.lm_head.weight"] = t(1000, 64)
    layout = discover_layout({"m.safetensors": h})

    assert layout.prefix == "language_model.model."
    assert layout.num_layers == 30
    assert layout.lm_head == "language_model.lm_head.weight"
    assert layout.tie_word_embeddings is False
    assert layout.is_multimodal


def test_prefixo_qwen2_5_vl():
    """Dimensões e decoder na raiz; só a torre é descartada."""
    layout = discover_layout({"m.safetensors": decoder("model.", 36) | visao_qwen()})

    assert layout.prefix == "model."
    assert layout.num_layers == 36
    assert layout.lm_head is None
    assert layout.tie_word_embeddings is True
    assert layout.needs_extraction


def test_prefixo_qwen3_vl_aninhado():
    h = decoder("model.language_model.", 36)
    h |= {f"model.visual.blocks.{i}.attn.qkv.weight": t(64, 64) for i in range(3)}
    layout = discover_layout({"m.safetensors": h})

    assert layout.prefix == "model.language_model."
    assert layout.rename["model.language_model.layers.0.mlp.up_proj.weight"] == \
        "model.layers.0.mlp.up_proj.weight"


def test_lm_head_fora_do_prefixo():
    """O SmolVLM grava o decoder em `model.text_model.` e o lm_head na raiz."""
    h = decoder("model.text_model.", 24)
    h["lm_head.weight"] = t(1000, 64)
    h["model.vision_model.encoder.layers.0.self_attn.out_proj.weight"] = t(64, 64)
    layout = discover_layout({"m.safetensors": h})

    assert layout.prefix == "model.text_model."
    assert layout.lm_head == "lm_head.weight"
    assert layout.rename["lm_head.weight"] == "lm_head.weight"


def test_decoder_ja_canonico_nao_precisa_extracao():
    layout = discover_layout({"m.safetensors": decoder("model.", 32)})

    assert layout.is_multimodal is False
    assert layout.needs_extraction is False


# ------------------------------------------------------- falsos positivos


@pytest.mark.parametrize("torre,ruido", [(visao_deepseek, "aligner.layers."),
                                          (visao_qwen, "visual.blocks.")])
def test_torre_e_projetor_nao_sao_candidatos(torre, ruido):
    layout = discover_layout({"m.safetensors": decoder("language_model.model.", 4) | torre()})

    assert layout.prefix == "language_model.model."
    assert any(n.startswith(ruido) for n in layout.dropped)


def test_dois_decoders_e_ambiguidade():
    h = decoder("language_model.model.", 4) | decoder("drafter.model.", 4)
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert "mais de um decoder" in e.value.message
    assert "language_model.model." in e.value.message and "drafter.model." in e.value.message


def test_sem_candidato_nomeia_o_que_faltou():
    """Phi-3 funde q/k/v e gate/up; a mensagem precisa dizer isso."""
    fundido = {"self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
               "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
               "input_layernorm.weight", "post_attention_layernorm.weight"}
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": decoder("model.", 32, sufixos=fundido)})

    assert "model." in e.value.message
    assert "self_attn.q_proj.weight" in e.value.message
    assert "Phi-3" in e.value.hint


def test_checkpoint_sem_camadas():
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": {"transformer.wte.weight": t(1000, 64)}})

    assert "layers.N" in e.value.message


# ------------------------------------------------------ integridade do decoder


def test_camadas_com_buraco():
    h = decoder("model.", 4)
    for chave in [k for k in h if ".layers.2." in k]:
        del h[chave]
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert "buracos" in e.value.message and "2" in e.value.message


def test_camadas_heterogeneas():
    """Uma camada com cross-attention denuncia intercalamento de tipos."""
    h = decoder("model.", 8)
    h["model.layers.3.cross_attn.q_proj.weight"] = t(64, 64)
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert "camada 3" in e.value.message and "cross_attn" in e.value.message


@pytest.mark.parametrize("faltante", ["embed_tokens.weight", "norm.weight"])
def test_decoder_incompleto(faltante):
    h = decoder("model.", 4)
    del h[f"model.{faltante}"]
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert faltante in e.value.message


def test_tensor_inesperado_sob_o_prefixo():
    h = decoder("model.text_model.", 4)
    h["model.text_model.adapter.weight"] = t(64, 64)
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert "inesperado" in e.value.message and "adapter" in e.value.message


def test_dois_lm_head():
    h = decoder("language_model.model.", 4)
    h["language_model.lm_head.weight"] = t(1000, 64)
    h["outro.lm_head.weight"] = t(1000, 64)
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"m.safetensors": h})

    assert "mais de um lm_head" in e.value.message


def test_tensor_repetido_entre_shards():
    h = decoder("model.", 4)
    with pytest.raises(UnsupportedArchitecture) as e:
        discover_layout({"a.safetensors": h, "b.safetensors": {"model.norm.weight": t(64)}})

    assert "a.safetensors" in e.value.message and "b.safetensors" in e.value.message


# ------------------------------------------------------------- contabilidade


def test_contagem_separa_texto_de_visao():
    h = decoder("language_model.model.", 2, hidden=64, vocab=1000) | visao_qwen()
    layout = discover_layout({"m.safetensors": h})

    esperado_visao = 3 * 3 * 64 * 64
    assert layout.dropped_params == esperado_visao
    assert layout.kept_params == 1000 * 64 + 64 + 2 * len(SIGNATURE) * 64 * 64
    assert set(layout.kept).isdisjoint(layout.dropped)


def test_metadata_nao_conta_como_tensor():
    h = decoder("model.", 2)
    layout = discover_layout({"m.safetensors": {"__metadata__": {"format": "pt"}} | h})

    assert "__metadata__" not in layout.kept and "__metadata__" not in layout.dropped


def test_renomeacao_leva_tudo_para_a_raiz_canonica():
    h = decoder("language_model.model.", 3)
    h["language_model.lm_head.weight"] = t(1000, 64)
    layout = discover_layout({"m.safetensors": h})

    assert all(n.startswith("model.") or n == "lm_head.weight"
               for n in layout.rename.values())
    assert layout.rename["language_model.model.embed_tokens.weight"] == \
        "model.embed_tokens.weight"
    assert layout.rename["language_model.model.norm.weight"] == "model.norm.weight"

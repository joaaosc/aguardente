"""Emissão do config.json text-only e conferência do que foi escrito."""

import json

import pytest

from aguardente.arch import Arch
from aguardente.errors import UnsupportedArchitecture
from aguardente.textonly import (APPLE_FALLBACK, arch_from_shapes, build_config,
                                 copy_auxiliary, discover_layout,
                                 emitted_model_type, strip_multimodal,
                                 supported_model_types, verify_extraction)
from tests.test_textonly_arch import checkpoint, deepseek
from tests.test_textonly_rewrite import escrever, layout_de

LAYOUT_DEEPSEEK = discover_layout({"m.safetensors": deepseek()})

ARCH = Arch(hidden_size=4096, intermediate_size=11008, num_hidden_layers=30,
            num_attention_heads=32, num_key_value_heads=32, head_dim=128,
            vocab_size=102400, tie_word_embeddings=False)


# ------------------------------------------------- registro do exportador


def test_lista_embutida_reflete_o_registro_da_apple():
    aceitos, do_pacote = supported_model_types()
    if not do_pacote:
        assert aceitos == APPLE_FALLBACK
    assert {"mistral", "qwen2", "qwen3", "gemma3_text", "phi3"} <= aceitos


@pytest.mark.parametrize("origem,alvo", [
    ("llama", "mistral"),            # DeepSeek-VL, LLaVA, SmolVLM
    ("qwen2_5_vl", "qwen2"),         # dimensões no topo, sem sub-config
    ("qwen2_vl", "qwen2"),
    ("qwen3_vl_text", "qwen3"),
    ("gemma3", "gemma3_text"),
    ("gemma3_text", "gemma3_text"),  # já é o nome do registro
    ("qwen2", "qwen2"),              # InternVL3
    ("mistral", "mistral"),
])
def test_equivalencia_de_model_type(origem, alvo):
    assert emitted_model_type(origem) == alvo


@pytest.mark.parametrize("desconhecido", ["internlm2", "falcon", "molmo", "phi4mm"])
def test_arquitetura_fora_do_registro_e_recusada(desconhecido):
    with pytest.raises(UnsupportedArchitecture) as e:
        emitted_model_type(desconhecido)

    assert desconhecido in e.value.message
    assert "mistral" in e.value.hint


# -------------------------------------------------------- limpeza do config


@pytest.mark.parametrize("chave", ["vision_config", "aligner_config", "visual",
                                    "image_token_id", "vision_start_token_id",
                                    "auto_map", "architectures"])
def test_chaves_de_outro_componente_saem(chave):
    assert chave not in strip_multimodal({"hidden_size": 8, chave: "x"})


def test_mrope_e_removido():
    """Em texto puro os três eixos recebem a mesma posição: M-RoPE vira RoPE."""
    limpo = strip_multimodal({"rope_scaling": {"type": "mrope",
                                               "mrope_section": [16, 24, 24]}})
    assert "rope_scaling" not in limpo


def test_mrope_interleaved_do_qwen3_vl_e_removido():
    limpo = strip_multimodal({"rope_scaling": {"mrope_interleaved": True,
                                               "mrope_section": [24, 20, 20],
                                               "rope_type": "default"}})
    assert "rope_scaling" not in limpo


@pytest.mark.parametrize("escala", [
    {"factor": 8.0, "rope_type": "linear"},              # Gemma 3
    {"factor": 2.0, "rope_type": "dynamic", "type": "dynamic"},  # InternVL3
])
def test_escala_de_rope_verdadeira_sobrevive(escala):
    """Descartar o fator do Gemma 3 encurtaria o contexto do modelo em silêncio."""
    assert strip_multimodal({"rope_scaling": escala})["rope_scaling"] == escala


# ------------------------------------------------------------ build_config


def test_deepseek_vira_mistral_com_as_dimensoes_das_formas():
    cfg = {"model_type": "multi_modality",
           "language_config": {"model_type": "llama", "num_hidden_layers": 30,
                               "vocab_size": 102400, "max_position_embeddings": 16384},
           "vision_config": {"cls": "HybridVisionTower"},
           "aligner_config": {"cls": "MlpProjector"}}
    r = build_config(cfg, ARCH, LAYOUT_DEEPSEEK)

    assert r.source_type == "llama" and r.emitted_type == "mistral"
    assert r.config["architectures"] == ["MistralForCausalLM"]
    assert r.config["hidden_size"] == 4096 and r.config["intermediate_size"] == 11008
    assert r.config["max_position_embeddings"] == 16384
    assert "vision_config" not in r.config and "language_config" not in r.config


def test_defaults_de_classe_sao_materializados():
    """O language_config do DeepSeek-VL omite rope_theta e rms_norm_eps."""
    cfg = {"model_type": "multi_modality",
           "language_config": {"model_type": "llama", "num_hidden_layers": 30}}
    r = build_config(cfg, ARCH, LAYOUT_DEEPSEEK)

    assert set(r.defaults_used) == {"rope_theta", "rms_norm_eps", "hidden_act",
                                    "initializer_range"}
    assert r.config["rope_theta"] == 10000.0 and r.config["rms_norm_eps"] == 1e-6


def test_valor_declarado_nao_e_sobrescrito_por_default():
    cfg = {"model_type": "multi_modality",
           "language_config": {"model_type": "llama", "rope_theta": 500000.0}}
    r = build_config(cfg, ARCH, LAYOUT_DEEPSEEK)

    assert r.config["rope_theta"] == 500000.0
    assert "rope_theta" not in r.defaults_used


def test_dimensoes_vencem_o_config_de_origem():
    """As formas são a verdade: um config que discorde delas é corrigido."""
    cfg = {"model_type": "llama", "hidden_size": 1, "num_hidden_layers": 999}
    r = build_config(cfg, ARCH, LAYOUT_DEEPSEEK)

    assert r.config["hidden_size"] == 4096 and r.config["num_hidden_layers"] == 30


# --------------------------------------------------------- arquivos auxiliares


def test_processador_de_imagem_nao_e_copiado(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir(); dst.mkdir()
    (src / "tokenizer.json").write_text("{}")
    (src / "preprocessor_config.json").write_text("{}")
    (src / "processor_config.json").write_text("{}")
    copiados = copy_auxiliary(src, dst)

    assert "tokenizer.json" in copiados
    assert not (dst / "preprocessor_config.json").exists()
    assert not (dst / "processor_config.json").exists()


def test_processor_class_sai_do_tokenizer_config(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir(); dst.mkdir()
    (src / "tokenizer_config.json").write_text(json.dumps(
        {"processor_class": "VLChatProcessor", "model_max_length": 16384}))
    copy_auxiliary(src, dst)
    saida = json.loads((dst / "tokenizer_config.json").read_text())

    assert "processor_class" not in saida
    assert saida["model_max_length"] == 16384


# ------------------------------------------------------------- verificação


DECODER = checkpoint("model.", hidden=64, inter=128, camadas=2, vocab=32,
                     q_out=64, kv_out=64)


def arch_do_decoder():
    headers = {"m.safetensors": DECODER}
    return arch_from_shapes(headers, discover_layout(headers), cfg={"head_dim": 64})


def escrever_decoder(destino):
    """Grava o decoder de referência com formas coerentes com a contagem."""
    destino.mkdir(parents=True, exist_ok=True)
    tensores = {}
    for nome, meta in DECODER.items():
        forma = tuple(meta["shape"])
        n = 1
        for d in forma:
            n *= d
        tensores[nome] = ("F16", forma, b"\x00" * (n * 2))
    escrever(destino / "model.safetensors", tensores)
    return destino


def test_verificacao_aceita_extracao_correta(tmp_path):
    d = escrever_decoder(tmp_path / "texto")

    layout = verify_extraction(d, arch_do_decoder())
    assert layout.prefix == "model." and layout.num_layers == 2


def test_verificacao_recusa_contagem_divergente(tmp_path):
    d = escrever_decoder(tmp_path / "texto")
    arch = arch_do_decoder()

    with pytest.raises(UnsupportedArchitecture) as e:
        verify_extraction(d, arch.with_(num_hidden_layers=3))

    assert "camadas" in e.value.message or "parâmetros" in e.value.message


def test_verificacao_recusa_diretorio_vazio(tmp_path):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    with pytest.raises(UnsupportedArchitecture) as e:
        verify_extraction(vazio, arch_do_decoder())

    assert "safetensors" in e.value.message


def test_token_especial_acima_do_vocabulario(tmp_path):
    d = escrever_decoder(tmp_path / "texto")
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"added_tokens_decoder": {"99": {"content": "<image>"}}}))
    with pytest.raises(UnsupportedArchitecture) as e:
        verify_extraction(d, arch_do_decoder())

    assert "vocabulário de 32" in e.value.message

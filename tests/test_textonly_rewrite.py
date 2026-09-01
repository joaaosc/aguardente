"""Reescrita dos safetensors mantendo só o decoder de texto.

A extração copia faixas de bytes e reconstrói o cabeçalho, sem interpretar os
tensores. Os arquivos destes testes são escritos byte a byte, no formato real —
u64 com o tamanho do cabeçalho, JSON, buffer de dados —, sem framework algum.
"""

import json
import struct

import pytest

from aguardente.errors import UnsupportedArchitecture
from aguardente.textonly import (SIGNATURE, discover_layout, encode_header,
                                 extract_weights)


def escrever(caminho, tensores, *, folga=0):
    """Grava um safetensors a partir de {nome: (dtype, shape, bytes)}."""
    entradas, corpo = {}, b""
    for nome, (dtype, shape, dados) in tensores.items():
        entradas[nome] = {"dtype": dtype, "shape": list(shape),
                          "data_offsets": [len(corpo), len(corpo) + len(dados)]}
        corpo += dados
    cab = json.dumps({"__metadata__": {"format": "pt"}} | entradas,
                     separators=(",", ":")).encode() + b" " * folga
    if sobra := (8 + len(cab)) % 8:
        cab += b" " * (8 - sobra)
    caminho.write_bytes(struct.pack("<Q", len(cab)) + cab + corpo)


def ler(caminho):
    """Devolve {nome: (dtype, shape, bytes)} de um safetensors gravado."""
    bruto = caminho.read_bytes()
    n = struct.unpack("<Q", bruto[:8])[0]
    header = json.loads(bruto[8:8 + n])
    dados = bruto[8 + n:]
    return {k: (v["dtype"], tuple(v["shape"]),
                dados[v["data_offsets"][0]:v["data_offsets"][1]])
            for k, v in header.items() if k != "__metadata__"}, header


def decoder_tensores(prefixo, camadas, *, dtype="F16", semente=b"\x11"):
    t = {f"{prefixo}embed_tokens.weight": (dtype, (8, 4), semente * 64),
         f"{prefixo}norm.weight": (dtype, (4,), semente * 8)}
    for i in range(camadas):
        for j, sufixo in enumerate(sorted(SIGNATURE)):
            t[f"{prefixo}layers.{i}.{sufixo}"] = (
                dtype, (4, 4), bytes([(i * 20 + j) % 251]) * 32)
    return t


@pytest.fixture
def vlm(tmp_path):
    """Checkpoint em dois shards: o primeiro misto, o segundo só texto."""
    origem = tmp_path / "teacher"
    origem.mkdir()
    misto = {"vision_model.tower.weight": ("F16", (16, 4), b"\xaa" * 128),
             "aligner.proj.weight": ("F16", (4, 4), b"\xbb" * 32)}
    misto |= decoder_tensores("language_model.model.", 1)
    escrever(origem / "model-00001-of-00002.safetensors", misto)

    so_texto = {f"language_model.model.layers.{i}.{s}": ("F16", (4, 4),
                bytes([(i * 20 + j) % 251]) * 32)
                for i in (1, 2) for j, s in enumerate(sorted(SIGNATURE))}
    so_texto["language_model.lm_head.weight"] = ("F16", (8, 4), b"\xcc" * 64)
    escrever(origem / "model-00002-of-00002.safetensors", so_texto)
    return origem


def layout_de(origem):
    headers = {}
    for p in sorted(origem.glob("*.safetensors")):
        _, header = ler(p)
        headers[p.name] = {k: v for k, v in header.items() if k != "__metadata__"}
    return discover_layout(headers)


def saida(destino):
    """Une os tensores de todos os shards de saída."""
    juntos = {}
    for p in sorted(destino.glob("*.safetensors")):
        conteudo, _ = ler(p)
        juntos |= conteudo
    return juntos


# ------------------------------------------------------------ ida e volta


def test_bytes_dos_tensores_mantidos_sao_identicos(vlm, tmp_path):
    origem_tensores = {}
    for p in sorted(vlm.glob("*.safetensors")):
        conteudo, _ = ler(p)
        origem_tensores |= conteudo

    layout = layout_de(vlm)
    extract_weights(vlm, tmp_path / "texto", layout)
    resultado = saida(tmp_path / "texto")

    for original, canonico in layout.rename.items():
        assert resultado[canonico] == origem_tensores[original], canonico


def test_torre_e_projetor_nao_aparecem(vlm, tmp_path):
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm))
    nomes = set(saida(tmp_path / "texto"))

    assert not any(n.startswith(("vision_model", "aligner")) for n in nomes)
    assert "model.embed_tokens.weight" in nomes and "lm_head.weight" in nomes


def test_nomes_saem_na_raiz_canonica(vlm, tmp_path):
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm))

    for nome in saida(tmp_path / "texto"):
        assert nome.startswith("model.") or nome == "lm_head.weight"


def test_dtype_e_forma_preservados(tmp_path):
    """bfloat16 não tem representação em NumPy; a cópia por bytes não se importa."""
    origem = tmp_path / "t"
    origem.mkdir()
    escrever(origem / "model.safetensors", decoder_tensores("model.", 2, dtype="BF16"))
    extract_weights(origem, tmp_path / "texto", layout_de(origem))

    for dtype, forma, _ in saida(tmp_path / "texto").values():
        assert dtype == "BF16" and len(forma) in (1, 2)


# ------------------------------------------------------------- formato


def test_cabecalho_alinhado_em_oito(vlm, tmp_path):
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm))

    for p in (tmp_path / "texto").glob("*.safetensors"):
        n = struct.unpack("<Q", p.read_bytes()[:8])[0]
        assert (8 + n) % 8 == 0


def test_metadata_declara_o_formato(vlm, tmp_path):
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm))
    p = sorted((tmp_path / "texto").glob("*.safetensors"))[0]
    _, header = ler(p)

    assert header["__metadata__"] == {"format": "pt"}


def test_encode_header_respeita_o_alinhamento():
    for extra in range(9):
        bruto = encode_header({"a" * (extra + 1): {"dtype": "F16", "shape": [1],
                                                    "data_offsets": [0, 2]}})
        n = struct.unpack("<Q", bruto[:8])[0]
        assert (8 + n) % 8 == 0 and len(bruto) == 8 + n


# --------------------------------------------------------------- shards


def test_arquivo_unico_dispensa_indice(tmp_path):
    origem = tmp_path / "t"
    origem.mkdir()
    escrever(origem / "model.safetensors", decoder_tensores("model.", 2))
    r = extract_weights(origem, tmp_path / "texto", layout_de(origem))

    assert r.shards == ("model.safetensors",)
    assert not (tmp_path / "texto" / "model.safetensors.index.json").exists()


def test_limite_de_shard_reparte_a_saida(vlm, tmp_path):
    r = extract_weights(vlm, tmp_path / "texto", layout_de(vlm), shard_limit=200)

    assert len(r.shards) > 1
    assert all(n.startswith("model-") and f"of-{len(r.shards):05d}" in n
               for n in r.shards)


def test_indice_cobre_todos_os_tensores(vlm, tmp_path):
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm), shard_limit=200)
    indice = json.loads((tmp_path / "texto" / "model.safetensors.index.json").read_text())

    assert set(indice["weight_map"]) == set(saida(tmp_path / "texto"))
    assert indice["metadata"]["total_size"] == sum(
        len(b) for _, _, b in saida(tmp_path / "texto").values())
    for arquivo in set(indice["weight_map"].values()):
        assert (tmp_path / "texto" / arquivo).is_file()


def test_tensor_maior_que_o_limite_nao_trava(tmp_path):
    """Um tensor sozinho acima do limite ocupa um shard só para ele."""
    origem = tmp_path / "t"
    origem.mkdir()
    escrever(origem / "model.safetensors", decoder_tensores("model.", 2))
    r = extract_weights(origem, tmp_path / "texto", layout_de(origem), shard_limit=8)

    assert r.tensors == len(saida(tmp_path / "texto"))


# ------------------------------------------------ reaproveitamento de shard


def test_shard_so_de_texto_e_reaproveitado(vlm, tmp_path):
    """O segundo shard é 100% decoder: só o cabeçalho muda, os dados não são copiados."""
    r = extract_weights(vlm, tmp_path / "texto", layout_de(vlm), reuse_shards=True)

    assert r.reused == ("model-00002-of-00002.safetensors",)
    assert r.copied_bytes < r.total_bytes
    assert r.saved_bytes > 0
    assert not (vlm / "model-00002-of-00002.safetensors").exists()


def test_reaproveitamento_preserva_os_bytes(vlm, tmp_path):
    antes = {}
    for p in sorted(vlm.glob("*.safetensors")):
        conteudo, _ = ler(p)
        antes |= conteudo
    layout = layout_de(vlm)
    extract_weights(vlm, tmp_path / "texto", layout, reuse_shards=True)
    depois = saida(tmp_path / "texto")

    for original, canonico in layout.rename.items():
        assert depois[canonico] == antes[original], canonico


def test_sem_reuse_a_origem_fica_intacta(vlm, tmp_path):
    antes = sorted(p.name for p in vlm.glob("*.safetensors"))
    extract_weights(vlm, tmp_path / "texto", layout_de(vlm))

    assert sorted(p.name for p in vlm.glob("*.safetensors")) == antes


def test_cabecalho_que_nao_cabe_cai_para_copia(tmp_path):
    """Sem folga no cabeçalho original, reescrevê-lo no lugar é impossível."""
    origem = tmp_path / "t"
    origem.mkdir()
    # Prefixo curto: a renomeação para `model.` não encurta nada, e o nome do
    # lm_head cresce de `lm_head.weight` para o mesmo, sem sobra.
    tensores = decoder_tensores("m.", 1)
    escrever(origem / "model.safetensors", tensores)
    r = extract_weights(origem, tmp_path / "texto", layout_de(origem),
                        reuse_shards=True)

    assert r.reused == ()
    assert r.copied_bytes == r.total_bytes


# ----------------------------------------------------------- atomicidade


def test_falha_no_meio_nao_deixa_destino(vlm, tmp_path, monkeypatch):
    import aguardente.textonly as mod

    def explodir(*a, **k):
        raise OSError("disco cheio")

    monkeypatch.setattr(mod, "_copiar", explodir)
    with pytest.raises(OSError):
        extract_weights(vlm, tmp_path / "texto", layout_de(vlm))

    assert not (tmp_path / "texto").exists()
    assert not (tmp_path / "texto.parcial").exists()


def test_origem_sem_safetensors(vlm, tmp_path):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    with pytest.raises(UnsupportedArchitecture) as e:
        extract_weights(vazio, tmp_path / "texto", layout_de(vlm))

    assert "safetensors" in e.value.message

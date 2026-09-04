"""Viabilidade de conversão e destilação em outra máquina, a partir de RAM/disco.

`_secao_outra_maquina` não toca a máquina atual: as contas são as mesmas do
plano local (`Budget`, `plan_for_target`), só que sobre uma `Machine`
descrita pelos números informados, e não pela detectada por `sysctl`.
"""

import pytest

from aguardente.arch import Arch
from aguardente.budget import GB, Machine
from aguardente import effort
from aguardente.cli import _secao_outra_maquina
from aguardente.probe import ModelProbe

NIVEL = effort.get(effort.DEFAULT)

# Qwen3-4B real: piso de poda medido em 0,96 B, teto de treino de uma
# máquina de 24 GB medido em 1,21 B — os dois números vêm de sessões
# anteriores e servem de referência cruzada para os testes abaixo.
QWEN3_4B = Arch(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                vocab_size=151936, tie_word_embeddings=True)


def probe_de(arch, params):
    return ModelProbe(ref="org/modelo", arch=arch, model_type="qwen3",
                      stored_params=params)


@pytest.fixture
def espiao(monkeypatch):
    """Silencia a saída e devolve (rótulo, valor) de cada ui.field chamado."""
    import aguardente.cli as cli

    registro = []
    monkeypatch.setattr(cli.ui, "header", lambda *a, **k: None)
    monkeypatch.setattr(cli.ui, "blank", lambda *a, **k: None)
    monkeypatch.setattr(cli.ui, "field",
                        lambda rotulo, valor, **k: registro.append((rotulo, valor, k)))
    return registro


def valor_de(registro, rotulo):
    return next(v for r, v, _ in registro if r == rotulo)


def nota_de(registro, rotulo):
    return next(k.get("note") for r, v, k in registro if r == rotulo)


# ------------------------------------------------------------- Machine.other


def test_machine_other_converte_gb_para_bytes():
    m = Machine.other(ram_gb=24, disk_gb=480)
    assert m.ram_bytes == 24 * GB
    assert m.free_disk_bytes == 480 * GB


def test_machine_other_assume_apple_silicon():
    assert Machine.other(ram_gb=8, disk_gb=20).arm64 is True


# -------------------------------------------------------- teto de treino


def test_maquina_24gb_bate_com_a_referencia_medida(espiao):
    """1,21 B é o número documentado para uma máquina de 24 GB — regressão."""
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None, NIVEL)
    teto = valor_de(espiao, "teto de treino")
    assert "1,2" in teto or "1.2" in teto


# ------------------------------------------------------------ destilação


def test_maquina_grande_permite_destilacao(espiao):
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None, NIVEL)
    assert valor_de(espiao, "destilação (recuperação)") == "viável"
    assert not any(r == "piso de poda" for r, _, _ in espiao)


def test_maquina_pequena_recusa_destilacao_e_diz_o_motivo(espiao):
    """0,96 B é o piso de poda documentado do Qwen3-4B — regressão cruzada."""
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 8, 20, None, NIVEL)

    assert valor_de(espiao, "destilação (recuperação)") == "inviável"
    piso = valor_de(espiao, "piso de poda")
    assert "961" in piso or "0.96" in piso or "0,96" in piso
    motivo = nota_de(espiao, "destilação (recuperação)")
    assert "piso de poda" in motivo and "skip-recover" in motivo


def test_target_params_explicito_e_respeitado(espiao):
    """Um alvo pedido pelo usuário vale mais que o teto da máquina."""
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480,
                         500_000_000, NIVEL)
    # 500 M é menor que o piso de poda (0,96 B): mesmo numa máquina grande,
    # esse alvo específico não é alcançável.
    assert valor_de(espiao, "destilação (recuperação)") == "inviável"


# ------------------------------------------------------------------ disco


def test_disco_insuficiente_e_recusado_com_o_que_falta(espiao):
    # Qwen3-4B em fp16 já são ~7,5 GB só de download; 3 GB não bastam.
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 3, None, NIVEL)

    assert valor_de(espiao, "conversão (poda + export)") == "inviável"
    motivo = nota_de(espiao, "conversão (poda + export)")
    assert "faltam" in motivo


def test_disco_generoso_e_aceito(espiao):
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None, NIVEL)
    assert valor_de(espiao, "conversão (poda + export)") == "viável"


def test_estimativa_de_disco_soma_download_bundle_e_logits(espiao):
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None, NIVEL)
    nota = nota_de(espiao, "disco no pico (estimado)")
    assert "download" in nota and "bundle" in nota
    # Os logits costumam ser o maior artefato do pipeline: omiti-los anunciava
    # "viável" para máquinas onde a execução real aborta por falta de disco.
    assert "logits" in nota and NIVEL.name in nota
    # Modelo de texto puro: a etapa de extração não deveria aparecer.
    assert "extração" not in nota


def test_esforco_maior_aumenta_o_pico_de_disco(espiao):
    """O disco dos logits cresce com o nível, e a conta precisa refletir isso."""
    from aguardente import effort

    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None,
                         effort.get("low"))
    baixo = nota_de(espiao, "disco no pico (estimado)")
    espiao.clear()
    _secao_outra_maquina(QWEN3_4B, probe_de(QWEN3_4B, 4_022_468_096), 24, 480, None,
                         effort.get("max"))
    alto = nota_de(espiao, "disco no pico (estimado)")
    assert baixo != alto


def test_modelo_multimodal_soma_a_extracao(espiao, monkeypatch):
    """Um checkpoint multimodal precisa de espaço extra para o decoder extraído."""
    import aguardente.cli as cli

    p = probe_de(QWEN3_4B, 4_022_468_096)
    monkeypatch.setattr(type(p), "is_multimodal", property(lambda self: True))
    monkeypatch.setattr(type(p), "text_params", property(lambda self: 3_000_000_000))
    _secao_outra_maquina(QWEN3_4B, p, 24, 480, None, NIVEL)

    nota = nota_de(espiao, "disco no pico (estimado)")
    assert "extração" in nota

"""Testes do comando doctor, incluindo o modo detalhado."""

import argparse

import pytest

from aguardente import cli
from aguardente.preflight import CheckResult, Status

FALHA = CheckResult("Xcode", Status.FAIL, "Command Line Tools ativo",
                    hint="execute xcode-select -s ...",
                    debug="$ xcodebuild -version  (exit=1)\n  requires Xcode")
OK = CheckResult("aria2c", Status.OK, "1.37.0")


AVISO = CheckResult("stack do pipeline", Status.WARN, "faltam: torch",
                    hint="aguardente install")


@pytest.fixture
def espiao(monkeypatch):
    """Silencia a saída e registra as chamadas relevantes de ui."""
    registro = {"step": [], "error": [], "done": []}
    for nome in ("title", "blank", "explain", "check", "rule"):
        monkeypatch.setattr(cli.ui, nome, lambda *a, **k: None)
    monkeypatch.setattr(cli.ui, "step", lambda t, **k: registro["step"].append(t))
    monkeypatch.setattr(cli.ui, "error", lambda m, **k: registro["error"].append(k))
    monkeypatch.setattr(cli.ui, "done", lambda m, **k: registro["done"].append(m))
    return registro


def _doctor(verbose, resultados, monkeypatch):
    monkeypatch.setattr(cli, "run_all", lambda **kw: resultados)
    return cli.cmd_doctor(argparse.Namespace(verbose=verbose))


def test_falha_retorna_codigo_1(espiao, monkeypatch):
    assert _doctor(False, [OK, FALHA], monkeypatch) == 1


def test_ambiente_saudavel_retorna_0(espiao, monkeypatch):
    assert _doctor(False, [OK], monkeypatch) == 0


def test_modo_normal_oculta_saida_bruta(espiao, monkeypatch):
    _doctor(False, [OK, FALHA], monkeypatch)
    assert espiao["step"] == []


def test_modo_verbose_emite_cada_linha_do_debug(espiao, monkeypatch):
    _doctor(True, [OK, FALHA], monkeypatch)
    assert espiao["step"] == ["$ xcodebuild -version  (exit=1)", "  requires Xcode"]


def test_modo_normal_anuncia_o_verbose(espiao, monkeypatch):
    _doctor(False, [OK, FALHA], monkeypatch)
    assert "--verbose" in espiao["error"][0]["action"]


def test_modo_verbose_nao_reanuncia_a_si_mesmo(espiao, monkeypatch):
    _doctor(True, [OK, FALHA], monkeypatch)
    assert "--verbose" not in espiao["error"][0]["action"]


def test_falha_sem_debug_nao_anuncia_o_verbose(espiao, monkeypatch):
    """Falhas como aria2c ausente não têm saída bruta a mostrar."""
    sem_debug = CheckResult("aria2c", Status.FAIL, "ausente", hint="brew install aria2")
    _doctor(False, [sem_debug], monkeypatch)
    assert "--verbose" not in espiao["error"][0]["action"]


def test_parser_aceita_a_flag():
    assert cli.build_parser().parse_args(["doctor", "--verbose"]).verbose is True
    assert cli.build_parser().parse_args(["doctor", "-v"]).verbose is True
    assert cli.build_parser().parse_args(["doctor"]).verbose is False


def test_aviso_nao_e_contado_como_aprovacao(espiao, monkeypatch):
    """Contar WARN como sucesso descrevia como íntegro um ambiente que `run` bloqueia."""
    assert _doctor(False, [OK, AVISO], monkeypatch) == 0
    resumo = espiao["done"][0]
    assert resumo.startswith("1 de 2 verificações passaram")
    assert "1 aviso(s)" in resumo and "stack do pipeline" in resumo


def test_ambiente_sem_avisos_nao_menciona_avisos(espiao, monkeypatch):
    _doctor(False, [OK], monkeypatch)
    assert "aviso" not in espiao["done"][0]


def test_saida_da_ui_e_capturavel(capsys):
    """Regressão: `sys.stdout` fixado no import impedia a captura em teste."""
    from aguardente import ui
    ui.note("mensagem visível")
    assert "mensagem visível" in capsys.readouterr().out


def test_ui_respeita_o_destino_informado(tmp_path):
    from aguardente import ui
    destino = tmp_path / "saida.txt"
    with destino.open("w") as fh:
        ui.note("para o arquivo", out=fh)
    assert "para o arquivo" in destino.read_text()

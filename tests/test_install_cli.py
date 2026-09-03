"""Testes do instalador de dependências do pipeline."""

import argparse
import subprocess

import pytest

from aguardente import cli
from aguardente.errors import AguardenteError
from aguardente.preflight import CheckResult, Status


def _silence_ui(monkeypatch):
    for name in ("title", "blank", "explain", "command", "error", "done"):
        monkeypatch.setattr(cli.ui, name, lambda *args, **kwargs: None)


def test_parser_expoe_install():
    args = cli.build_parser().parse_args(["install"])
    assert args.func is cli.cmd_install


def test_extrai_apenas_requisitos_da_extra_pipeline(monkeypatch):
    monkeypatch.setattr(cli, "requires", lambda name: [
        "pytest>=8; extra == 'dev'",
        "torch>=2.8; extra == 'pipeline'",
        'coreai-opt; extra == "pipeline"',
    ])
    assert cli._pipeline_requirements() == ["torch>=2.8", "coreai-opt"]


def test_install_usa_o_python_do_executavel(monkeypatch):
    _silence_ui(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(cli, "_pipeline_requirements", lambda: ["torch>=2.8", "coreai-opt"])
    monkeypatch.setattr(cli, "check_pipeline_deps", lambda **_: CheckResult(
        "stack do pipeline", Status.OK, "torch 2.9.0"))
    # `cmd_install` também confere os requisitos fora do Python depois de
    # instalar, e essas verificações também usam subprocess: guardar todas as
    # chamadas e olhar a primeira mantém a asserção sobre a instalação em si.
    chamadas = []

    def fake_run(command, **kwargs):
        chamadas.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.cmd_install(argparse.Namespace()) == 0
    command, kwargs = chamadas[0]
    assert command == [
        "/opt/bin/uv", "pip", "install", "--python", cli.sys.executable,
        "torch>=2.8", "coreai-opt",
    ]
    assert kwargs == {"check": False}


def test_install_propaga_falha_do_uv(monkeypatch):
    _silence_ui(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(cli, "_pipeline_requirements", lambda: ["torch"])
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 7))
    assert cli.cmd_install(argparse.Namespace()) == 7


def test_install_exige_uv(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(AguardenteError, match="uv não foi encontrado"):
        cli.cmd_install(argparse.Namespace())


def test_install_rejeita_python_incompativel(monkeypatch):
    monkeypatch.setattr(cli, "check_python", lambda: CheckResult(
        "Python", Status.FAIL, "3.14.7 free-threaded", "Use Python 3.12 convencional."))
    with pytest.raises(AguardenteError, match="não é compatível"):
        cli.cmd_install(argparse.Namespace())


def test_install_avisa_sobre_requisitos_fora_do_python(monkeypatch, capsys):
    """Instalar os pacotes Python não basta: sem aria2c o primeiro download falha."""
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(cli, "_pipeline_requirements", lambda: ["torch>=2.8"])
    monkeypatch.setattr(cli, "check_pipeline_deps", lambda **_: CheckResult(
        "stack do pipeline", Status.OK, "torch 2.9.0"))
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(cli, "_requisitos_externos", lambda: [
        CheckResult("aria2", Status.FAIL, "não encontrado", "brew install aria2")])

    assert cli.cmd_install(argparse.Namespace()) == 0
    saida = " ".join(capsys.readouterr().out.split())
    assert "aria2" in saida and "aguardente doctor" in saida


def test_install_nao_avisa_quando_o_ambiente_esta_completo(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(cli, "_pipeline_requirements", lambda: ["torch>=2.8"])
    monkeypatch.setattr(cli, "check_pipeline_deps", lambda **_: CheckResult(
        "stack do pipeline", Status.OK, "torch 2.9.0"))
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(cli, "_requisitos_externos", lambda: [
        CheckResult("aria2", Status.OK, "1.37.0")])
    monkeypatch.setattr(cli, "_e_ambiente_de_uv_tool", lambda: False)

    assert cli.cmd_install(argparse.Namespace()) == 0
    assert "aguardente doctor" not in capsys.readouterr().out


def test_deteccao_de_ambiente_uv_tool(monkeypatch):
    """Num ambiente `uv tool`, o que `uv pip install` grava some no próximo upgrade."""
    monkeypatch.setattr(cli.sys, "prefix", "/Users/x/.local/share/uv/tools/aguardente")
    assert cli._e_ambiente_de_uv_tool() is True
    monkeypatch.setattr(cli.sys, "prefix", "/Users/x/Projects/aguardente/.venv")
    assert cli._e_ambiente_de_uv_tool() is False


def test_comando_equivalente_grava_as_dependencias_no_recibo():
    assert cli._com_extras(["torch>=2.8", "coreai-opt"]) == [
        "--with", "torch>=2.8", "--with", "coreai-opt", "aguardente"]

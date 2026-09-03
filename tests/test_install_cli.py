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
    monkeypatch.setattr(cli, "check_pipeline_deps", lambda: CheckResult(
        "stack do pipeline", Status.OK, "torch 2.9.0"))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.cmd_install(argparse.Namespace()) == 0
    assert captured["command"] == [
        "/opt/bin/uv", "pip", "install", "--python", cli.sys.executable,
        "torch>=2.8", "coreai-opt",
    ]
    assert captured["kwargs"] == {"check": False}


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

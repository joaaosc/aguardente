"""Testes de tradução de falhas do coreai-build em erros acionáveis."""

import subprocess

import pytest

import aguardente.export as ex
from aguardente.errors import AguardenteError

CMD = ["xcrun", "coreai-build", "compile", "m.aimodel"]


def _falha(monkeypatch, exc, hint=None):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(ex, "toolchain_hint", lambda: hint)


def test_xcrun_ausente_usa_hint_do_toolchain(monkeypatch):
    _falha(monkeypatch, FileNotFoundError(), hint="rode xcode-select -s ...")
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=1)
    assert e.value.message == "xcrun não encontrado"
    assert e.value.hint == "rode xcode-select -s ..."


def test_xcrun_ausente_sem_diagnostico_tem_hint_padrao(monkeypatch):
    _falha(monkeypatch, FileNotFoundError())
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=1)
    assert "Xcode 27+" in e.value.hint


def test_timeout_vira_erro_com_comando_reproduzivel(monkeypatch):
    _falha(monkeypatch, subprocess.TimeoutExpired(CMD, 3600))
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=3600)
    assert "excedeu o tempo limite de 3600s" in e.value.message
    assert "xcrun coreai-build compile" in e.value.hint


def test_utilitario_ausente_sugere_metal_toolchain(monkeypatch):
    _falha(monkeypatch, subprocess.CalledProcessError(
        1, CMD, stderr='xcrun: error: unable to find utility "coreai-build"'))
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=1)
    assert "MetalToolchain" in e.value.hint


def test_toolchain_errado_tem_precedencia_sobre_metal_toolchain(monkeypatch):
    """Com o CLT ativo, baixar o Metal Toolchain não resolveria nada."""
    _falha(monkeypatch, subprocess.CalledProcessError(
        1, CMD, stderr='xcrun: error: unable to find utility "coreai-build"'),
        hint="selecione o Xcode")
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=1)
    assert e.value.hint == "selecione o Xcode"


def test_erro_do_proprio_compile_preserva_stderr(monkeypatch):
    _falha(monkeypatch, subprocess.CalledProcessError(
        2, CMD, stderr="operação não suportada no alvo"))
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="compile", timeout=1)
    assert "operação não suportada no alvo" in e.value.message


def test_erro_sem_stderr_reporta_codigo(monkeypatch):
    _falha(monkeypatch, subprocess.CalledProcessError(3, CMD, stderr=""))
    with pytest.raises(AguardenteError) as e:
        ex._run_coreai(CMD, acao="inspect", timeout=1)
    assert "código de saída 3" in e.value.message

"""Testes da rede de segurança do CLI e do descarte de execuções anteriores."""

import errno
import json

import pytest

from aguardente import cli
from aguardente.errors import AguardenteError


def falhar_com(monkeypatch, exc):
    def _raiser(args):
        raise exc
    monkeypatch.setattr(cli, "cmd_doctor", _raiser)


def erro(capsys) -> str:
    return capsys.readouterr().err


# ------------------------------------------------------------ rede de segurança


def test_memory_error_vira_mensagem_acionavel(monkeypatch, capsys):
    falhar_com(monkeypatch, MemoryError())
    assert cli.main(["doctor"]) == 1
    saida = erro(capsys)
    assert "memória insuficiente" in saida and "--target-params" in saida


def test_modulo_ausente_aponta_a_instalacao(monkeypatch, capsys):
    falhar_com(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert cli.main(["doctor"]) == 1
    saida = erro(capsys)
    assert "dependência ausente: torch" in saida and "aguardente install" in saida


def test_disco_cheio_e_identificado_pelo_errno(monkeypatch, capsys):
    falhar_com(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
    assert cli.main(["doctor"]) == 1
    saida = erro(capsys)
    assert "disco cheio" in saida and "retomada" in saida


def test_permissao_negada_e_identificada(monkeypatch, capsys):
    falhar_com(monkeypatch, PermissionError(errno.EACCES, "Permission denied", "/run"))
    assert cli.main(["doctor"]) == 1
    assert "permissão negada" in erro(capsys)


def test_falha_inesperada_menciona_a_variavel_de_depuracao(monkeypatch, capsys):
    falhar_com(monkeypatch, ValueError("algo estranho"))
    assert cli.main(["doctor"]) == 1
    saida = erro(capsys)
    assert "falha inesperada: ValueError" in saida and cli.DEBUG_ENV in saida


def test_traceback_so_aparece_sob_a_variavel(monkeypatch, capsys):
    falhar_com(monkeypatch, ValueError("algo estranho"))
    monkeypatch.delenv(cli.DEBUG_ENV, raising=False)
    cli.main(["doctor"])
    assert "Traceback" not in erro(capsys)

    monkeypatch.setenv(cli.DEBUG_ENV, "1")
    cli.main(["doctor"])
    assert "Traceback" in erro(capsys)


def test_erro_do_pacote_mantem_o_formato_anterior(monkeypatch, capsys):
    falhar_com(monkeypatch, AguardenteError("falhou", hint="tente assim"))
    assert cli.main(["doctor"]) == 1
    saida = erro(capsys)
    assert "erro: falhou" in saida and "tente assim" in saida


def test_interrupcao_mantem_codigo_130(monkeypatch, capsys):
    falhar_com(monkeypatch, KeyboardInterrupt())
    assert cli.main(["doctor"]) == 130
    assert "interrompida" in erro(capsys)


# --------------------------------------------------------------------- restart


def test_restart_remove_estado_e_artefatos(tmp_path):
    (tmp_path / "state.json").write_text("{}")
    for nome in ("teacher", "logits", "ckpt"):
        (tmp_path / nome).mkdir()
        (tmp_path / nome / "peso.bin").write_bytes(b"x")
    (tmp_path / "notas.txt").write_text("preservar")

    removidos = cli._discard_run_dir(tmp_path)

    assert set(removidos) == {"state.json", "teacher/", "logits/", "ckpt/"}
    assert not (tmp_path / "teacher").exists()
    # O que não pertence ao pipeline permanece: --restart não limpa o diretório inteiro.
    assert (tmp_path / "notas.txt").exists()


def test_restart_em_diretorio_limpo_nao_remove_nada(tmp_path):
    assert cli._discard_run_dir(tmp_path) == []


def test_development_launcher_preserves_cli_failure_exit_code(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/dev_cli.py"
    result = subprocess.run([sys.executable, str(script), "status", "-o", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 1


# ------------------------------------------------------------------- parser


@pytest.mark.parametrize("flag", ["--restart", "--allow-oversized"])
def test_flags_novas_existem_no_run(flag):
    args = cli.build_parser().parse_args(["run", "org/m", "-o", "run", flag])
    assert getattr(args, flag.lstrip("-").replace("-", "_")) is True


def test_flags_novas_tem_padrao_falso():
    args = cli.build_parser().parse_args(["run", "org/m", "-o", "run"])
    assert args.restart is False and args.allow_oversized is False


def test_erro_vira_evento_em_modo_json(monkeypatch, capsys):
    """Sem isto, a GUI só recebe 'processo encerrou com código 1'."""
    def explode(_args):
        raise AguardenteError("o alvo excede o teto de treino",
                              hint="Use --target-params menor.")

    parser = cli.build_parser()
    args = parser.parse_args(["run", "org/m", "-o", "run", "--json"])
    monkeypatch.setattr(args, "func", explode, raising=False)
    monkeypatch.setattr(cli, "build_parser", lambda: _ParserFixo(args))

    assert cli.main([]) == 1
    saida = capsys.readouterr()
    evento = json.loads([l for l in saida.out.splitlines() if l.strip()][-1])
    assert evento["ev"] == "error"
    assert "teto de treino" in evento["msg"]
    assert "--target-params" in evento["hint"]


def test_erro_sem_json_nao_emite_evento(monkeypatch, capsys):
    def explode(_args):
        raise AguardenteError("falhou")

    parser = cli.build_parser()
    args = parser.parse_args(["run", "org/m", "-o", "run"])
    monkeypatch.setattr(args, "func", explode, raising=False)
    monkeypatch.setattr(cli, "build_parser", lambda: _ParserFixo(args))

    assert cli.main([]) == 1
    assert capsys.readouterr().out == ""


class _ParserFixo:
    def __init__(self, args):
        self._args = args

    def parse_args(self, _argv=None):
        return self._args

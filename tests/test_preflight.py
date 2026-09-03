"""Testes de diagnóstico diferenciado do preflight."""

import subprocess

import pytest

from aguardente import preflight as pf

CLT = "/Library/Developer/CommandLineTools"
XCODE = "/Applications/Xcode.app/Contents/Developer"


def fake_probe(monkeypatch, table, *, dirs_existem=True):
    """Substitui _probe por uma tabela {primeiro argumento: Run}.

    `dirs_existem` simula a presença do diretório ativo do toolchain, que na
    máquina de teste raramente coincide com o caminho encenado.
    """
    def _probe(*cmd, timeout=10):
        return table.get(cmd[0], pf.Run(pf.NOT_FOUND))
    monkeypatch.setattr(pf, "_probe", _probe)
    monkeypatch.setattr(pf, "_isdir", lambda path: bool(path) and dirs_existem)
    monkeypatch.delenv(pf.DEVELOPER_DIR_ENV, raising=False)


def test_xcode_clt_selecionado(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, CLT),
        "xcodebuild": pf.Run(1, err="xcode-select: error: tool 'xcodebuild' requires Xcode, "
                                     "but active developer directory is a CommandLineTools instance"),
    })
    r = pf.check_xcode()
    assert r.status is pf.Status.FAIL
    assert "Command Line Tools" in r.detail
    assert pf.SELECT_XCODE in r.hint


def test_xcode_licenca_nao_aceita(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, XCODE),
        "xcodebuild": pf.Run(69, err="You have not agreed to the Xcode license agreements."),
    })
    r = pf.check_xcode()
    assert r.detail == "licença não aceita"
    assert "-license accept" in r.hint


def test_xcode_first_launch(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, XCODE),
        "xcodebuild": pf.Run(1, err="Xcode needs to install additional components."),
    })
    assert pf.check_xcode().detail == "instalação incompleta"
    assert "-runFirstLaunch" in pf.check_xcode().hint


def test_xcode_ausente(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(pf.NOT_FOUND),
                             "xcodebuild": pf.Run(pf.NOT_FOUND)})
    r = pf.check_xcode()
    assert r.detail == "ausente"
    assert "App Store" in r.hint


def test_xcode_timeout(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, XCODE),
                             "xcodebuild": pf.Run(pf.TIMEOUT)})
    assert pf.check_xcode().detail == "sem resposta"


def test_xcode_versao_antiga(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, XCODE),
                             "xcodebuild": pf.Run(0, "Xcode 26.2\nBuild version 26C100")})
    r = pf.check_xcode()
    assert r.status is pf.Status.FAIL
    assert r.detail.startswith("26")


def test_xcode_ok(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, XCODE),
                             "xcodebuild": pf.Run(0, "Xcode 27.0\nBuild version 27A100")})
    assert pf.check_xcode().status is pf.Status.OK


def test_coreai_build_culpa_do_clt(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, CLT),
        "xcrun": pf.Run(1, err="xcrun: error: unable to find utility \"coreai-build\""),
    })
    r = pf.check_coreai_build()
    assert r.detail == "Command Line Tools ativo"
    assert pf.SELECT_XCODE in r.hint


def test_coreai_build_metal_toolchain_ausente(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, XCODE),
        "xcrun": pf.Run(1, err="xcrun: error: unable to find utility \"coreai-build\""),
    })
    r = pf.check_coreai_build()
    assert r.detail == "ausente"
    assert "MetalToolchain" in r.hint


def test_coreai_build_ok(monkeypatch):
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, XCODE),
        "xcrun": pf.Run(0, "/usr/bin/coreai-build"),
    })
    r = pf.check_coreai_build()
    assert r.status is pf.Status.OK


@pytest.mark.parametrize("exc,code", [(FileNotFoundError(), pf.NOT_FOUND),
                                      (subprocess.TimeoutExpired("x", 1), pf.TIMEOUT)])
def test_probe_converte_excecoes(monkeypatch, exc, code):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(subprocess, "run", boom)
    assert pf._probe("qualquer").code == code


def test_run_preserva_contrato_anterior(monkeypatch):
    fake_probe(monkeypatch, {"ok": pf.Run(0, "saída"), "ruim": pf.Run(2, "x", "erro")})
    assert pf._run("ok") == "saída"
    assert pf._run("ruim") is None


def test_licenca_detectada_por_codigo_de_saida_sem_texto():
    """Locale traduzido: o exit code 69 identifica a licença mesmo sem stderr em inglês."""
    detalhe, hint = pf._diagnose_xcodebuild(
        pf.Run(pf.LICENSE_EXIT_CODE, err="Você não concordou com os termos."), XCODE)
    assert detalhe == "licença não aceita"
    assert "-license accept" in hint


def test_falha_generica_expoe_primeira_linha_do_stderr(monkeypatch):
    monkeypatch.setattr(pf, "_isdir", lambda path: True)
    detalhe, hint = pf._diagnose_xcodebuild(
        pf.Run(70, err="erro interno inesperado\nsegunda linha"), XCODE)
    assert "erro interno inesperado" in detalhe
    assert "segunda linha" not in detalhe


def test_diretorio_ativo_inexistente_e_causa_propria(monkeypatch):
    """Após mover ou atualizar o Xcode, o caminho selecionado deixa de existir."""
    fake_probe(monkeypatch, {
        "xcode-select": pf.Run(0, "/Volumes/Antigo/Xcode.app/Contents/Developer"),
        "xcodebuild": pf.Run(1, err="unable to locate developer directory"),
    }, dirs_existem=False)
    r = pf.check_xcode()
    assert r.detail == "diretório ativo inexistente"
    assert pf.SELECT_XCODE in r.hint


def test_developer_dir_tem_precedencia_sobre_xcode_select(monkeypatch):
    """Com DEVELOPER_DIR apontando para o Xcode, não há o que corrigir no xcode-select."""
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, CLT)})
    monkeypatch.setenv(pf.DEVELOPER_DIR_ENV, XCODE)
    assert pf._developer_dir() == XCODE
    assert pf.toolchain_hint() is None


def test_hint_de_clt_oferece_alternativa_sem_sudo(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, CLT)})
    assert pf.DEVELOPER_DIR_ENV in pf.toolchain_hint()


def test_disco_abaixo_do_piso_vira_aviso(monkeypatch, tmp_path):
    """Sem estimativa concreta, o espaço livre é comparado com um piso mínimo."""
    from aguardente.budget import GB, Machine
    monkeypatch.setattr(pf.Machine, "detect",
                        classmethod(lambda cls, path="/": Machine(24 * GB, 5 * GB, 8, True)))
    disco = [r for r in pf.check_resources(path=tmp_path) if r.name == "disco"][0]
    assert disco.status is pf.Status.WARN
    assert "mínimo recomendado" in disco.detail


def test_stack_do_pipeline_pode_ser_promovida_a_bloqueio():
    """WARN é informativo no doctor e requisito em run: a diferença é de contexto."""
    aviso = pf.CheckResult("stack do pipeline", pf.Status.WARN, "faltam: torch")
    ok = pf.CheckResult("Python", pf.Status.OK, "3.12.14")
    assert pf.blocking([aviso, ok]) == []
    assert pf.blocking([aviso, ok], warn_as_fail=("stack do pipeline",)) == [aviso]


def test_python_free_threaded_e_rejeitado(monkeypatch):
    monkeypatch.setattr(pf.sysconfig, "get_config_var", lambda name: 1)
    result = pf.check_python()
    assert result.status is pf.Status.FAIL
    assert "free-threaded" in result.detail
    assert "Python 3.12" in result.hint


def test_dependencia_presente_mas_incompativel_vira_aviso(monkeypatch):
    monkeypatch.setattr(pf.importlib.util, "find_spec", lambda name: object())

    def fake_import(name):
        if name == "coreai_opt":
            raise RuntimeError("ABI incompatível")
        return type("Module", (), {"__version__": "2.9.0"})()

    monkeypatch.setattr(pf.importlib, "import_module", fake_import)
    result = pf.check_pipeline_deps()
    assert result.status is pf.Status.WARN
    assert "coreai_opt não carrega" in result.detail
    assert "aguardente install" in result.hint
    assert "RuntimeError" in result.debug


def test_probe_forca_locale_c(monkeypatch):
    """As mensagens de erro precisam vir em inglês para o diagnóstico por texto funcionar."""
    capturado = {}

    def fake_run(cmd, **kw):
        capturado.update(kw.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pf._probe("xcodebuild", "-version")
    assert capturado["LC_ALL"] == "C" and capturado["LANG"] == "C"


def test_trace_registra_comando_e_estado():
    r = pf.Run(1, err="boom", cmd=("xcodebuild", "-version"))
    assert "$ xcodebuild -version" in r.trace() and "exit=1" in r.trace()
    assert "binário não encontrado" in pf.Run(pf.NOT_FOUND, cmd=("x",)).trace()
    assert "tempo esgotado" in pf.Run(pf.TIMEOUT, cmd=("x",)).trace()


def test_trace_limita_saida_a_oito_linhas():
    r = pf.Run(1, err="\n".join(f"linha {i}" for i in range(50)), cmd=("x",))
    assert len(r.trace().splitlines()) == 9  # cabeçalho + 8


def test_falha_preenche_debug_para_modo_verbose(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, CLT),
                             "xcodebuild": pf.Run(1, err="requires Xcode", cmd=("xcodebuild",))})
    assert "xcodebuild" in pf.check_xcode().debug


def test_sucesso_nao_preenche_debug(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, XCODE),
                             "xcodebuild": pf.Run(0, "Xcode 27.0")})
    assert pf.check_xcode().debug is None


def test_toolchain_hint_aponta_xcode_quando_clt_ativo(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, CLT)})
    assert pf.SELECT_XCODE in pf.toolchain_hint()


def test_toolchain_hint_silencioso_com_xcode_correto(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(0, XCODE)})
    assert pf.toolchain_hint() is None


def test_toolchain_hint_quando_xcode_select_falha(monkeypatch):
    fake_probe(monkeypatch, {"xcode-select": pf.Run(pf.NOT_FOUND)})
    assert "Xcode 27+" in pf.toolchain_hint()

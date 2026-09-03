"""Verificação de pré-requisitos do ambiente de execução."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from .budget import GB, Machine

# Códigos sintéticos para falhas que não produzem código de saída do processo.
NOT_FOUND = -1
TIMEOUT = -2

LICENSE_EXIT_CODE = 69

XCODE_DEV_DIR = "/Applications/Xcode.app/Contents/Developer"
SELECT_XCODE = f"sudo xcode-select -s {XCODE_DEV_DIR}"
DEVELOPER_DIR_ENV = "DEVELOPER_DIR"
# Alternativa para máquinas gerenciadas, em que `sudo xcode-select` é bloqueado.
SEM_SUDO = (f"Sem privilégio de administrador, exporte {DEVELOPER_DIR_ENV}={XCODE_DEV_DIR} "
            "no shell antes de executar o pipeline.")

# Piso de espaço livre para uma execução típica, na ausência de estimativa real.
MIN_FREE_DISK_BYTES = 30 * GB


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    hint: str | None = None
    debug: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL


@dataclass(frozen=True, slots=True)
class Run:
    """Resultado bruto de um comando externo, preservando stderr e código de saída."""

    code: int
    out: str = ""
    err: str = ""
    cmd: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.code == 0

    def trace(self) -> str:
        """Transcrição do comando para exibição em modo detalhado."""
        rotulo = {NOT_FOUND: "binário não encontrado", TIMEOUT: "tempo esgotado"}
        estado = rotulo.get(self.code, f"exit={self.code}")
        linhas = [f"$ {' '.join(self.cmd)}  ({estado})"]
        linhas += [f"  {ln}" for ln in (self.err or self.out).splitlines()[:8]]
        return "\n".join(linhas)


def _probe(*cmd: str, timeout: int = 10) -> Run:
    """Executa um comando sem levantar exceções, preservando o motivo da falha."""
    # LC_ALL=C fixa as mensagens de erro em inglês, para que o diagnóstico por
    # texto não dependa do idioma configurado na máquina do usuário.
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        return Run(NOT_FOUND, cmd=cmd)
    except subprocess.TimeoutExpired:
        return Run(TIMEOUT, cmd=cmd)
    except OSError as e:
        return Run(NOT_FOUND, err=str(e), cmd=cmd)
    return Run(r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip(), cmd=cmd)


def _run(*cmd: str, timeout: int = 10) -> str | None:
    r = _probe(*cmd, timeout=timeout)
    return r.out if r.ok else None


def _version_tuple(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    import re
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    return tuple(int(g) for g in m.groups() if g) if m else ()


def _developer_dir() -> str | None:
    """Diretório ativo do toolchain.

    `DEVELOPER_DIR` tem precedência sobre `xcode-select -p` tanto para `xcrun`
    quanto para `xcodebuild`; consultar apenas o segundo faria o diagnóstico
    sugerir uma correção desnecessária.
    """
    env = os.environ.get(DEVELOPER_DIR_ENV, "").strip()
    if env:
        return env
    r = _probe("xcode-select", "-p")
    return r.out if r.ok and r.out else None


def _isdir(path: str | None) -> bool:
    """Existência do diretório do toolchain, isolada para permitir simulação em teste."""
    return bool(path) and os.path.isdir(path)


def _clt_active(dev_dir: str | None, err: str = "") -> bool:
    """Indica que o diretório ativo são os Command Line Tools, e não o Xcode."""
    return "CommandLineTools" in (dev_dir or "") or "commandlinetools" in err.lower()


def check_macos() -> CheckResult:
    v = _version_tuple(_run("sw_vers", "-productVersion"))
    if platform.system() != "Darwin":
        return CheckResult("macOS", Status.FAIL, platform.system(),
                           "Core AI está disponível apenas no macOS.")
    if not v:
        return CheckResult("macOS", Status.WARN, "versão desconhecida")
    txt = ".".join(map(str, v))
    if v[0] < 27:
        return CheckResult("macOS", Status.FAIL, txt,
                           "Core AI exige macOS 27.0 ou superior.")
    return CheckResult("macOS", Status.OK, txt)


def _diagnose_xcodebuild(run: Run, dev_dir: str | None) -> tuple[str, str]:
    """Traduz uma falha de `xcodebuild` em (detalhe, instrução de correção)."""
    err = run.err.lower()
    if run.code == NOT_FOUND:
        return ("ausente",
                "Instale o Xcode 27+ pela App Store e selecione-o com "
                f"`{SELECT_XCODE}`.")
    if run.code == TIMEOUT:
        return ("sem resposta",
                "`xcodebuild -version` excedeu o tempo limite. Execute-o manualmente no "
                "Terminal, conclua o que for pedido e repita o diagnóstico.")
    if _clt_active(dev_dir, err):
        return ("Command Line Tools ativo",
                "Somente os Command Line Tools estão selecionados. Com o Xcode instalado, "
                f"execute `{SELECT_XCODE}`. {SEM_SUDO}")
    if run.code == LICENSE_EXIT_CODE or "licen" in err:
        return ("licença não aceita", "Execute `sudo xcodebuild -license accept`.")
    if "runfirstlaunch" in err or "first launch" in err or "additional components" in err:
        return ("instalação incompleta", "Execute `xcodebuild -runFirstLaunch`.")
    # Verificado depois dos sinais conclusivos acima: eles só chegam de um
    # xcodebuild que rodou, o que já implica um diretório ativo existente.
    if dev_dir and not _isdir(dev_dir):
        return ("diretório ativo inexistente",
                f"O toolchain selecionado ({dev_dir}) não existe — situação comum após "
                f"atualizar ou mover o Xcode. Execute `{SELECT_XCODE}`.")
    first = run.err.splitlines()[0] if run.err else f"código de saída {run.code}"
    return (f"falha: {first[:60]}",
            "Execute `xcodebuild -version` no Terminal para ver o erro completo.")


def check_xcode() -> CheckResult:
    dev_dir = _developer_dir()
    run = _probe("xcodebuild", "-version", timeout=30)
    if not run.ok:
        detail, hint = _diagnose_xcodebuild(run, dev_dir)
        return CheckResult("Xcode", Status.FAIL, detail, hint, debug=run.trace())
    v = _version_tuple(run.out.splitlines()[0] if run.out else None)
    if not v:
        return CheckResult("Xcode", Status.FAIL, "versão desconhecida",
                           "A saída de `xcodebuild -version` não pôde ser interpretada.")
    txt = ".".join(map(str, v))
    if v[0] < 27:
        return CheckResult("Xcode", Status.FAIL, txt, "Core AI exige Xcode 27.0 ou superior.")
    return CheckResult("Xcode", Status.OK, txt)


def check_coreai_build() -> CheckResult:
    run = _probe("xcrun", "--find", "coreai-build")
    if run.ok and run.out:
        ver = _run("xcrun", "coreai-build", "--version") or "?"
        return CheckResult("coreai-build", Status.OK, ver)
    trace = run.trace()
    if run.code == NOT_FOUND:
        return CheckResult("coreai-build", Status.FAIL, "xcrun ausente",
                           "As ferramentas de linha de comando do Xcode não estão instaladas. "
                           f"Instale o Xcode 27+ e execute `{SELECT_XCODE}`.", debug=trace)
    if run.code == TIMEOUT:
        return CheckResult("coreai-build", Status.FAIL, "sem resposta",
                           "`xcrun --find coreai-build` excedeu o tempo limite. Execute-o "
                           "manualmente no Terminal e repita o diagnóstico.", debug=trace)
    dev_dir = _developer_dir()
    if dev_dir and not _isdir(dev_dir):
        return CheckResult("coreai-build", Status.FAIL, "diretório ativo inexistente",
                           f"O toolchain selecionado ({dev_dir}) não existe. "
                           f"Execute `{SELECT_XCODE}`.", debug=trace)
    if _clt_active(dev_dir, run.err):
        return CheckResult("coreai-build", Status.FAIL, "Command Line Tools ativo",
                           "coreai-build acompanha o Xcode, não os Command Line Tools. "
                           f"Selecione o Xcode com `{SELECT_XCODE}` antes de baixar o "
                           f"Metal Toolchain. {SEM_SUDO}", debug=trace)
    return CheckResult("coreai-build", Status.FAIL, "ausente",
                       "Instale o Metal Toolchain: xcodebuild -downloadComponent MetalToolchain",
                       debug=trace)


def toolchain_hint() -> str | None:
    """Instrução corretiva quando o toolchain ativo não é capaz de resolver `xcrun`."""
    dev_dir = _developer_dir()
    if dev_dir is None:
        return ("As ferramentas de linha de comando do Xcode não respondem. Instale o "
                f"Xcode 27+ e execute `{SELECT_XCODE}`.")
    if not _isdir(dev_dir):
        return (f"O toolchain selecionado ({dev_dir}) não existe. Execute `{SELECT_XCODE}`.")
    if _clt_active(dev_dir):
        return (f"O diretório ativo é {dev_dir}, que não contém o Core AI toolchain. "
                f"Execute `{SELECT_XCODE}`. {SEM_SUDO}")
    return None


def check_aria2() -> CheckResult:
    """Verifica a disponibilidade do aria2c para download de arquivos."""
    path = shutil.which("aria2c")
    if not path:
        return CheckResult("aria2c", Status.FAIL, "ausente",
                           "Requisito para downloads retomáveis: brew install aria2")
    ver = _run("aria2c", "--version")
    first = ver.splitlines()[0] if ver else "?"
    return CheckResult("aria2c", Status.OK, first.replace("aria2 version ", ""))


def check_python() -> CheckResult:
    v = sys.version_info[:3]
    txt = ".".join(map(str, v))
    if sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        return CheckResult(
            "Python", Status.FAIL, f"{txt} free-threaded",
            "A stack Core AI ainda não suporta o ABI free-threaded. Use Python 3.12 convencional.",
        )
    if not ((3, 11) <= v[:2] < (3, 14)):
        return CheckResult("Python", Status.FAIL, txt,
                           "coreai-opt exige >=3.11,<3.14. Use `uv venv --python 3.12`.")
    return CheckResult("Python", Status.OK, txt)


def check_arch() -> CheckResult:
    m = platform.machine()
    if m != "arm64":
        return CheckResult("arquitetura", Status.FAIL, m,
                           "coreai-core só publica wheels para macOS arm64 — não há build Intel.")
    return CheckResult("arquitetura", Status.OK, m)


def check_pipeline_deps() -> CheckResult:
    """Verifica se os pacotes opcionais de execução do pipeline estão instalados."""
    missing = [m for m in ("torch", "transformers", "coreai_torch", "coreai_opt")
               if importlib.util.find_spec(m) is None]
    if missing:
        return CheckResult("stack do pipeline", Status.WARN, f"faltam: {', '.join(missing)}",
                           "Instale as dependências completas com: aguardente install")

    loaded = {}
    for module in ("torch", "transformers", "coreai_torch", "coreai_opt"):
        try:
            loaded[module] = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 — diagnóstico de dependência quebrada
            detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return CheckResult(
                "stack do pipeline", Status.WARN,
                f"{module} não carrega: {detail}",
                "Reinstale as dependências completas com: aguardente install",
                debug=f"{type(exc).__name__}: {exc}",
            )

    torch = loaded["torch"]
    return CheckResult("stack do pipeline", Status.OK, f"torch {torch.__version__}")


def check_resources(required_disk_bytes: int = 0, *, path: str | Path = "/") -> list[CheckResult]:
    """Mede RAM e espaço livre no volume indicado por `path`.

    Sem estimativa concreta, o espaço é comparado com um piso: uma execução
    típica precisa de dezenas de GB entre pesos, logits e artefatos.
    """
    m = Machine.detect(path)
    out = [CheckResult("RAM", Status.OK,
                       f"{m.ram_bytes / GB:.0f} GB (orçamento {m.usable_ram_bytes / GB:.0f} GB)")]
    free = m.free_disk_bytes / GB
    if required_disk_bytes:
        exigido, margem = required_disk_bytes, 1.3
        motivo = f"estimativa {exigido / GB:.0f} GB"
    else:
        exigido, margem = MIN_FREE_DISK_BYTES, 1.0
        motivo = f"mínimo recomendado {exigido / GB:.0f} GB"
    if m.free_disk_bytes < exigido * margem:
        out.append(CheckResult("disco", Status.WARN,
                               f"{free:.0f} GB livres, {motivo}",
                               "Espaço livre reduzido para os arquivos do modelo, os logits "
                               "pré-computados e os artefatos intermediários."))
    else:
        out.append(CheckResult("disco", Status.OK, f"{free:.0f} GB livres"))
    return out


def run_all(*, required_disk_bytes: int = 0, include_pipeline: bool = True,
            path: str | Path = "/") -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        check_macos, check_xcode, check_coreai_build, check_aria2,
        check_python, check_arch,
    ]
    if include_pipeline:
        checks.append(check_pipeline_deps)
    results = [c() for c in checks]
    results.extend(check_resources(required_disk_bytes, path=path))
    return results


def blocking(results: list[CheckResult], *,
             warn_as_fail: Sequence[str] = ()) -> list[CheckResult]:
    """Verificações que impedem o prosseguimento.

    `warn_as_fail` promove avisos a bloqueios pelo nome: a stack do pipeline é
    informativa no `doctor` e requisito em `run`, e a diferença é de contexto,
    não de gravidade.
    """
    nomes = set(warn_as_fail)
    return [r for r in results
            if r.status is Status.FAIL
            or (r.status is Status.WARN and r.name in nomes)]

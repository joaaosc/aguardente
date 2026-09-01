"""Verificação de pré-requisitos do ambiente de execução."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .budget import GB, Machine


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

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL


def _run(*cmd: str, timeout: int = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return r.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _version_tuple(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    import re
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    return tuple(int(g) for g in m.groups() if g) if m else ()


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


def check_xcode() -> CheckResult:
    out = _run("xcodebuild", "-version")
    v = _version_tuple(out.splitlines()[0] if out else None)
    if not v:
        return CheckResult("Xcode", Status.FAIL, "ausente",
                           "Instale o Xcode 27+ e execute `xcode-select --install`.")
    txt = ".".join(map(str, v))
    if v[0] < 27:
        return CheckResult("Xcode", Status.FAIL, txt, "Core AI exige Xcode 27.0 ou superior.")
    return CheckResult("Xcode", Status.OK, txt)


def check_coreai_build() -> CheckResult:
    path = _run("xcrun", "--find", "coreai-build")
    if not path:
        return CheckResult("coreai-build", Status.FAIL, "ausente",
                           "Instale o Metal Toolchain: xcodebuild -downloadComponent MetalToolchain")
    ver = _run("xcrun", "coreai-build", "--version") or "?"
    return CheckResult("coreai-build", Status.OK, ver)


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
    import importlib.util
    missing = [m for m in ("torch", "transformers", "coreai_torch", "coreai_opt")
               if importlib.util.find_spec(m) is None]
    if missing:
        return CheckResult("stack do pipeline", Status.WARN, f"faltam: {', '.join(missing)}",
                           "Instale as dependências completas com: uv pip install 'aguardente[pipeline]'")
    import torch  # noqa: PLC0415
    return CheckResult("stack do pipeline", Status.OK, f"torch {torch.__version__}")


def check_resources(required_disk_bytes: int = 0) -> list[CheckResult]:
    m = Machine.detect()
    out = [CheckResult("RAM", Status.OK,
                       f"{m.ram_bytes / GB:.0f} GB (orçamento {m.usable_ram_bytes / GB:.0f} GB)")]
    free = m.free_disk_bytes / GB
    if required_disk_bytes and m.free_disk_bytes < required_disk_bytes * 1.3:
        out.append(CheckResult("disco", Status.WARN,
                               f"{free:.0f} GB livres, estimativa {required_disk_bytes / GB:.0f} GB",
                               "Espaço livre reduzido para os arquivos do modelo e cache."))
    else:
        out.append(CheckResult("disco", Status.OK, f"{free:.0f} GB livres"))
    return out


def run_all(*, required_disk_bytes: int = 0, include_pipeline: bool = True) -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        check_macos, check_xcode, check_coreai_build, check_aria2,
        check_python, check_arch,
    ]
    if include_pipeline:
        checks.append(check_pipeline_deps)
    results = [c() for c in checks]
    results.extend(check_resources(required_disk_bytes))
    return results


def blocking(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.status is Status.FAIL]

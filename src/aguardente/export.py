"""Integração com coreai.llm.export para conversão em formato .aimodel."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import AguardenteError
from .preflight import toolchain_hint

DEFAULT_COMPRESSION = "4bit"          # preset macOS: int4 per-block(32), ~4.50 BPW
DEFAULT_PRECISION = "float16"


@dataclass(frozen=True, slots=True)
class ExportResult:
    bundle_dir: Path
    aimodel: Path | None
    metadata: dict[str, Any]

    @property
    def size_bytes(self) -> int:
        if not self.aimodel or not self.aimodel.exists():
            return 0
        return sum(f.stat().st_size for f in self.aimodel.rglob("*") if f.is_file())


def _resolve(name: str) -> list[str] | None:
    """Localiza o executável coreai.llm.export no PATH ou via módulo uv."""
    direct = shutil.which(name)
    if direct:
        return [direct]

    import importlib.util
    if importlib.util.find_spec("coreai_models") is not None and shutil.which("uv"):
        return ["uv", "run", name]
    return None


def available() -> bool:
    """Verifica se o exportador da Apple está disponível no ambiente."""
    return _resolve("coreai.llm.export") is not None


def build_command(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    platform: str = "macOS",
    compression: str = DEFAULT_COMPRESSION,
    compression_config: str | Path | None = None,
    compute_precision: str = DEFAULT_PRECISION,
    max_context_length: int | None = None,
    output_name: str | None = None,
    include_debug_info: bool = False,
    experimental: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Monta a lista de argumentos para invocação do coreai.llm.export."""
    base = _resolve("coreai.llm.export")
    if base is None:
        raise AguardenteError(
            "coreai.llm.export não encontrado — o exportador da Apple não está instalado",
            hint="Execute `aguardente install` para instalar o exportador oficial da Apple; "
                 "o pacote homônimo na PyPI é de terceiro e não serve.",
        )

    cmd = [*base, str(model_dir), "--platform", platform,
           "--compute-precision", compute_precision,
           "--output-dir", str(out_dir)]

    if compression_config:
        cmd += ["--compression-config", str(compression_config)]
    else:
        cmd += ["--compression", compression]

    if max_context_length:
        cmd += ["--max-context-length", str(max_context_length)]
    if output_name:
        cmd += ["--output-name", output_name]
    if include_debug_info:
        cmd.append("--include-debug-info")
    if experimental:
        # Necessário para diretórios locais de modelos
        cmd.append("--experimental")
    if overwrite:
        cmd.append("--overwrite")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_export(cmd: list[str], *, on_line: Callable[[str], None] | None = None,
               timeout: float | None = None) -> int:
    """Executa o comando de exportação repassando as linhas de saída."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_line:
                on_line(line.rstrip("\n"))
        return proc.wait(timeout=timeout)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        if proc.poll() is None:
            proc.kill()


def find_bundle(out_dir: str | Path) -> ExportResult:
    """Localiza o bundle gerado e carrega o arquivo metadata.json."""
    out = Path(out_dir)
    candidates = sorted(out.glob("**/metadata.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        aimodels = sorted(out.glob("**/*.aimodel"))
        if not aimodels:
            raise AguardenteError(f"nenhum bundle encontrado em {out}")
        return ExportResult(bundle_dir=aimodels[-1].parent, aimodel=aimodels[-1], metadata={})

    meta_path = candidates[-1]
    meta = json.loads(meta_path.read_text())
    bundle = meta_path.parent
    aimodels = sorted(bundle.glob("*.aimodel"))
    return ExportResult(bundle_dir=bundle, aimodel=aimodels[0] if aimodels else None,
                        metadata=meta)


# O tempo de compilação acompanha o tamanho do modelo: uma constante fixa
# bastava para os alvos atuais e não sobreviveria a um modelo maior.
COMPILE_TIMEOUT_MIN = 1800
COMPILE_SECONDS_PER_GB = 1200


def _compile_timeout(aimodel: Path) -> int:
    """Limite de tempo proporcional ao tamanho do artefato a compilar."""
    try:
        size = aimodel.stat().st_size
    except OSError:
        return COMPILE_TIMEOUT_MIN
    return int(max(COMPILE_TIMEOUT_MIN, size / (1024 ** 3) * COMPILE_SECONDS_PER_GB))


def _run_coreai(cmd: list[str], *, acao: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """Executa `xcrun coreai-build`, traduzindo falhas de toolchain em erro acionável."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError as e:
        raise AguardenteError(
            "xcrun não encontrado",
            hint=toolchain_hint() or "Instale o Xcode 27+ e as Command Line Tools.",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise AguardenteError(
            f"coreai-build {acao} excedeu o tempo limite de {timeout}s",
            hint="Execute o comando manualmente no Terminal para observar o progresso: "
                 f"{' '.join(cmd)}",
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        hint = toolchain_hint()
        if not hint and "unable to find utility" in stderr:
            hint = ("coreai-build não está disponível no toolchain ativo. Instale o Metal "
                    "Toolchain: xcodebuild -downloadComponent MetalToolchain")
        raise AguardenteError(
            f"coreai-build {acao} falhou: {stderr or f'código de saída {e.returncode}'}",
            hint=hint,
        ) from e


def inspect_asset(aimodel: str | Path, *, storage: bool = True, compute: bool = True,
                  ops: bool = True) -> dict[str, Any]:
    """Inspeciona o arquivo .aimodel utilizando `xcrun coreai-build inspect --json`."""
    cmd = ["xcrun", "coreai-build", "inspect", str(aimodel), "--json"]
    if storage:
        cmd.append("--storage")
    if compute:
        cmd.append("--compute")
    if ops:
        cmd.append("--ops")
    out = _run_coreai(cmd, acao="inspect", timeout=300)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise AguardenteError(f"saída de coreai-build inspect não é JSON: {e}") from e


def compile_aot(aimodel: str | Path, out_dir: str | Path, *, platform: str = "macOS",
                min_version: str = "27.0", timeout: int | None = None) -> list[Path]:
    """Compila o modelo AOT utilizando `xcrun coreai-build compile`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "coreai-build", "compile", str(aimodel),
           "--platform", platform, "--min-deployment-version", min_version,
           "--output", str(out)]
    _run_coreai(cmd, acao="compile",
                timeout=timeout if timeout is not None else _compile_timeout(Path(aimodel)))
    return sorted(out.glob("*.aimodelc"))

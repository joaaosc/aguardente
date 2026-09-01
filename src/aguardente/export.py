"""Ponte para o exportador da Apple.

`coreai.llm.export` já faz o caminho de PyTorch até `.aimodel` comprimido, com
presets, YAMLs de compressão e bundle com tokenizer. Reimplementá-lo seria
retrabalho — aqui só se monta a invocação e se lê o resultado.

Aceita um diretório local no formato transformers, que é exatamente o que a
etapa de poda grava.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import AguardenteError

DEFAULT_COMPRESSION = "4bit"          # preset macOS: int4 per-block(32), ~4,50 BPW
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
    """`coreai.llm.export` é um console script instalado pelo `coreai-models`.

    Procura o executável no PATH. Não cai para `uv run` quando o pacote não
    está instalado — isso produziria um `Failed to spawn` no meio do pipeline,
    depois de horas de trabalho, em vez de um erro claro no início.
    """
    direct = shutil.which(name)
    if direct:
        return [direct]

    import importlib.util
    if importlib.util.find_spec("coreai_models") is not None and shutil.which("uv"):
        # Instalado como biblioteca mas sem console script no PATH.
        return ["uv", "run", name]
    return None


def available() -> bool:
    """Se o exportador da Apple está instalado e invocável."""
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
    """Monta o argv. `--compression` e `--compression-config` são exclusivos."""
    base = _resolve("coreai.llm.export")
    if base is None:
        raise AguardenteError(
            "coreai.llm.export não encontrado — o exportador da Apple não está instalado",
            hint="uv pip install -e '.[pipeline]'  (traz coreai-models do GitHub; "
                 "o pacote homônimo no PyPI é de terceiro e NÃO serve)",
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
        # Exigido para modelos fora do registry — um diretório local nunca casa
        # com um preset, e a flag traz junto a obrigação de --compute-precision.
        cmd.append("--experimental")
    if overwrite:
        cmd.append("--overwrite")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_export(cmd: list[str], *, on_line: Callable[[str], None] | None = None,
               timeout: float | None = None) -> int:
    """Executa, repassando a saída linha a linha enquanto acontece."""
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
    """Localiza o bundle produzido e lê o `metadata.json` (schema 0.2)."""
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


def inspect_asset(aimodel: str | Path, *, storage: bool = True, compute: bool = True,
                  ops: bool = True) -> dict[str, Any]:
    """`xcrun coreai-build inspect --json`.

    O caminho mais barato para o relatório: tamanho, tipos de storage e compute,
    distribuição de operações e assinaturas de função — sem escrever Swift.
    """
    cmd = ["xcrun", "coreai-build", "inspect", str(aimodel), "--json"]
    if storage:
        cmd.append("--storage")
    if compute:
        cmd.append("--compute")
    if ops:
        cmd.append("--ops")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    except FileNotFoundError as e:
        raise AguardenteError(
            "xcrun não encontrado",
            hint="Instale o Xcode 27+ e as Command Line Tools.",
        ) from e
    except subprocess.CalledProcessError as e:
        raise AguardenteError(f"coreai-build inspect falhou: {e.stderr.strip()}") from e
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise AguardenteError(f"saída de coreai-build inspect não é JSON: {e}") from e


def compile_aot(aimodel: str | Path, out_dir: str | Path, *, platform: str = "macOS",
                min_version: str = "27.0") -> list[Path]:
    """`xcrun coreai-build compile` — um `.aimodelc` por arquitetura."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "coreai-build", "compile", str(aimodel),
           "--platform", platform, "--min-deployment-version", min_version,
           "--output", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)
    except subprocess.CalledProcessError as e:
        raise AguardenteError(f"coreai-build compile falhou: {e.stderr.strip()}") from e
    return sorted(out.glob("*.aimodelc"))

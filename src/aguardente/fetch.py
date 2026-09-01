"""Download de modelos do Hugging Face via aria2c."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

from .errors import AguardenteError

_HF = "https://huggingface.co"
_TIMEOUT = 30
_MAX_INDEX_BYTES = 8 * 1024 * 1024

_MODEL_ID = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$")


def validate_model_id(model_id: str) -> str:
    """Valida se o identificador do modelo segue a convenção `nome` ou `namespace/nome`."""
    if not isinstance(model_id, str) or not _MODEL_ID.match(model_id):
        raise AguardenteError(
            f"identificador de modelo inválido: {model_id!r}",
            hint="Use o formato `namespace/nome`, como `Qwen/Qwen3-4B`.",
        )
    return model_id

# Arquivos necessários para execução e carregamento do modelo
WANTED_EXACT = frozenset({
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "preprocessor_config.json",
})
WANTED_SUFFIX = (".safetensors",)

# Subdiretórios ou variantes excluídos do download
EXCLUDE_PARTS = ("onnx/", "openvino/", "coreml/", "gguf/", "/consolidated")


def safe_join(base: Path, relative: str) -> Path:
    """Resolve caminho relativo garantindo que permaneça dentro do diretório base."""
    if not relative or relative.strip() != relative:
        raise AguardenteError(f"caminho remoto inválido: {relative!r}")

    if "\\" in relative or ":" in relative:
        raise AguardenteError(
            f"caminho remoto com separador ou esquema inesperado: {relative!r}",
            hint="O caminho fornecido contém caracteres inválidos.",
        )

    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise AguardenteError(
            f"caminho remoto tenta sair do destino: {relative!r}",
            hint="Caminhos remotos não podem conter referências a diretórios pais.",
        )

    base_resolved = base.resolve()
    target = (base_resolved / candidate).resolve()
    if not target.is_relative_to(base_resolved):
        raise AguardenteError(
            f"caminho remoto resolve para fora do destino: {relative!r}",
            hint="O caminho resolvido precisa estar contido no diretório de destino.",
        )
    return base_resolved / candidate


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """Arquivo remoto identificado no repositório."""

    path: str
    size: int

    def __post_init__(self) -> None:
        safe_join(Path("/__validacao__"), self.path)

    def url(self, model_id: str, revision: str = "main") -> str:
        return f"{_HF}/{model_id}/resolve/{revision}/{self.path}"


@dataclass(frozen=True, slots=True)
class FetchPlan:
    model_id: str
    revision: str
    dest: Path
    files: tuple[RemoteFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def missing(self) -> tuple[RemoteFile, ...]:
        """Identifica arquivos ausentes ou incompletos no destino."""
        out = []
        for f in self.files:
            local = safe_join(self.dest, f.path)
            if not local.is_file() or (f.size and local.stat().st_size != f.size):
                out.append(f)
        return tuple(out)

    @property
    def pending_bytes(self) -> int:
        return sum(f.size for f in self.missing())


def _wanted(path: str) -> bool:
    """Verifica se o arquivo é necessário para o modelo e tem caminho seguro."""
    if not path or path.strip() != path:
        return False
    if path.startswith("/") or "\\" in path or ":" in path:
        return False
    if any(part == ".." for part in PurePosixPath(path).parts):
        return False
    if any(part in path for part in EXCLUDE_PARTS):
        return False
    name = path.rsplit("/", 1)[-1]
    return name in WANTED_EXACT or path.endswith(WANTED_SUFFIX)


def list_files(model_id: str, revision: str = "main") -> tuple[RemoteFile, ...]:
    """Lista os arquivos necessários do repositório no Hugging Face."""
    validate_model_id(model_id)
    url = f"{_HF}/api/models/{model_id}/tree/{revision}?recursive=1"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            bruto = r.read(_MAX_INDEX_BYTES + 1)
            if len(bruto) > _MAX_INDEX_BYTES:
                raise AguardenteError(
                    f"índice de {model_id} excede {_MAX_INDEX_BYTES // 1024 // 1024} MB",
                    hint="Resposta do índice excedeu o tamanho máximo permitido.",
                )
            entries = json.loads(bruto)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AguardenteError(
                f"acesso negado a {model_id}",
                hint="Modelo gated. Aceite a licença na página do Hugging Face "
                     "e rode `hf auth login`.",
            ) from e
        raise AguardenteError(f"HTTP {e.code} ao listar {model_id}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise AguardenteError(f"falha de rede ao listar {model_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise AguardenteError(f"índice de {model_id} não é JSON válido: {e}") from e

    if not isinstance(entries, list):
        raise AguardenteError(f"índice de {model_id} tem formato inesperado")

    files = []
    for e in entries:
        caminho = e.get("path")
        if not isinstance(caminho, str) or e.get("type") != "file":
            continue
        if not _wanted(caminho):
            continue
        size = e.get("size") or (e.get("lfs") or {}).get("size") or 0
        try:
            files.append(RemoteFile(path=caminho, size=int(size)))
        except AguardenteError:
            continue

    if not any(f.path.endswith(".safetensors") for f in files):
        raise AguardenteError(
            f"{model_id} não publica pesos em .safetensors",
            hint="Só o formato safetensors é suportado.",
        )
    return tuple(sorted(files, key=lambda f: f.path))


def plan_fetch(model_id: str, dest: str | Path, *, revision: str = "main") -> FetchPlan:
    return FetchPlan(model_id=model_id, revision=revision,
                     dest=Path(dest).expanduser(), files=list_files(model_id, revision))


def require_aria2() -> str:
    exe = shutil.which("aria2c")
    if not exe:
        raise AguardenteError(
            "aria2c não encontrado",
            hint="É requisito para downloads retomáveis. Instale com: brew install aria2",
        )
    return exe


def fetch(
    plan: FetchPlan,
    *,
    connections: int = 8,
    concurrent: int = 4,
    max_tries: int = 10,
    retry_wait: int = 5,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Executa o download dos arquivos do plano via aria2c."""
    for nome, valor, teto in (("connections", connections, 16),
                              ("concurrent", concurrent, 16),
                              ("max_tries", max_tries, 100)):
        if not isinstance(valor, int) or not 1 <= valor <= teto:
            raise AguardenteError(
                f"{nome} precisa ser um inteiro entre 1 e {teto} (recebido: {valor!r})")
    if not isinstance(retry_wait, int) or not 0 <= retry_wait <= 600:
        raise AguardenteError(
            f"retry_wait precisa ser um inteiro entre 0 e 600 (recebido: {retry_wait!r})")

    exe = require_aria2()
    pending = plan.missing()
    if not pending:
        return plan.dest

    plan.dest.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for f in pending:
        target = safe_join(plan.dest, f.path)
        lines.append(f.url(plan.model_id, plan.revision))
        lines.append(f"  dir={target.parent}")
        lines.append(f"  out={target.name}")
    input_file = plan.dest / ".aguardente-fetch.txt"
    input_file.write_text("\n".join(lines) + "\n")
    input_file.chmod(0o600)

    cmd = [
        exe,
        "--input-file", str(input_file),
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--max-concurrent-downloads={concurrent}",
        "--min-split-size=8M",
        f"--max-tries={max_tries}",
        f"--retry-wait={retry_wait}",
        "--timeout=60",
        "--connect-timeout=30",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--conditional-get=true",
        "--summary-interval=10",
        "--console-log-level=warn",
        "--download-result=hide",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and on_line:
                on_line(line)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        input_file.unlink(missing_ok=True)

    if code != 0:
        raise AguardenteError(
            f"aria2c saiu com código {code}",
            hint="O download é retomável: execute o mesmo comando novamente para continuar.",
        )

    still_missing = plan.missing()
    if still_missing:
        names = ", ".join(f.path for f in still_missing[:3])
        raise AguardenteError(
            f"{len(still_missing)} arquivo(s) incompletos após o download: {names}",
            hint="Execute o comando novamente para retomar os arquivos incompletos.",
        )
    return plan.dest


def fetch_model(
    model_id: str, dest: str | Path, *, revision: str = "main",
    on_line: Callable[[str], None] | None = None, **kw: Any,
) -> FetchPlan:
    """Planeja e executa o download do modelo."""
    plan = plan_fetch(model_id, dest, revision=revision)
    fetch(plan, on_line=on_line, **kw)
    return plan

"""Download de modelos do Hugging Face via aria2c."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

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
    sha256: str | None = None

    def __post_init__(self) -> None:
        safe_join(Path("/__validacao__"), self.path)
        if self.size < 0:
            raise AguardenteError(f"tamanho negativo para {self.path}")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise AguardenteError(f"checksum SHA-256 inválido para {self.path}")

    def url(self, model_id: str, revision: str = "main") -> str:
        return f"{_HF}/{model_id}/resolve/{revision}/{self.path}"


@dataclass(frozen=True, slots=True)
class FetchPlan:
    model_id: str
    revision: str
    dest: Path
    files: tuple[RemoteFile, ...]
    requested_revision: str = "main"

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def missing(self) -> tuple[RemoteFile, ...]:
        """Identifica arquivos ausentes ou incompletos no destino.

        O índice do Hugging Face nem sempre traz o tamanho — arquivos fora do
        LFS podem vir sem o campo. Quando isso acontecia, a comparação de
        tamanho era pulada por inteiro e qualquer arquivo existente passava por
        completo, inclusive um de zero byte deixado por um download abortado.
        Sem tamanho de referência, só dá para afirmar que um arquivo vazio está
        incompleto — e é o que se afirma.
        """
        out = []
        for f in self.files:
            local = safe_join(self.dest, f.path)
            if not local.is_file():
                out.append(f)
                continue
            tamanho_local = local.stat().st_size
            if f.size:
                if tamanho_local != f.size:
                    out.append(f)
            elif tamanho_local == 0:
                out.append(f)
            if f.sha256 and f not in out:
                from .inputs import file_identity
                if file_identity(local).lower() != f.sha256.lower():
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
    origin = urllib.parse.urlsplit(url)
    visited: set[str] = set()
    entries = []
    received = 0
    try:
        while url:
            if url in visited or len(visited) >= 100:
                raise AguardenteError(f"paginação inválida ou excessiva no índice de {model_id}")
            visited.add(url)
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
                bruto = r.read(_MAX_INDEX_BYTES - received + 1)
                received += len(bruto)
                if received > _MAX_INDEX_BYTES:
                    raise AguardenteError(
                        f"índice de {model_id} excede {_MAX_INDEX_BYTES // 1024 // 1024} MB",
                        hint="Resposta do índice excedeu o tamanho máximo permitido.",
                    )
                page = json.loads(bruto)
                if not isinstance(page, list):
                    raise AguardenteError(f"índice de {model_id} tem formato inesperado")
                entries.extend(page)
                link = r.headers.get("Link", "")
            following = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', link)
            url = urllib.parse.urljoin(url, following.group(1)) if following else ""
            if url:
                next_origin = urllib.parse.urlsplit(url)
                if (next_origin.scheme, next_origin.netloc, next_origin.path) != (origin.scheme, origin.netloc, origin.path):
                    raise AguardenteError("paginação do Hub mudou a origem ou revisão do índice")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            # Este endpoint lista a árvore de arquivos, e devolve 200 mesmo
            # para um repositório gated — o estado de acesso só entra na hora
            # de baixar o conteúdo de um arquivo, não na listagem. Um 401
            # aqui só acontece quando o identificador não existe.
            raise AguardenteError(
                f"{model_id} não encontrado no Hugging Face",
                hint="Confira o identificador (namespace/nome).",
            ) from e
        raise AguardenteError(f"HTTP {e.code} ao listar {model_id}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise AguardenteError(f"falha de rede ao listar {model_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise AguardenteError(f"índice de {model_id} não é JSON válido: {e}") from e

    files = []
    for e in entries:
        if not isinstance(e, dict):
            raise AguardenteError(f"entrada inválida no índice de {model_id}")
        caminho = e.get("path")
        if not isinstance(caminho, str) or e.get("type") != "file":
            continue
        if not _wanted(caminho):
            continue
        size = e.get("size") or (e.get("lfs") or {}).get("size") or 0
        files.append(RemoteFile(path=caminho, size=int(size), sha256=(e.get("lfs") or {}).get("oid")))

    if not any(f.path.endswith(".safetensors") for f in files):
        raise AguardenteError(
            f"{model_id} não publica pesos em .safetensors",
            hint="Só o formato safetensors é suportado.",
        )
    return tuple(sorted(files, key=lambda f: f.path))


def plan_fetch(model_id: str, dest: str | Path, *, revision: str = "main") -> FetchPlan:
    validate_model_id(model_id)
    dest = Path(dest).expanduser()
    provenance = dest / ".aguardente-source.json"
    saved_files = None
    if provenance.is_file():
        saved = json.loads(provenance.read_text())
        if saved["model"] != model_id or saved["requested_revision"] != revision:
            raise AguardenteError("diretório de download pertence a outro modelo/revisão",
                                  hint="Use outro destino para preservar os arquivos existentes.")
        commit = saved["commit"]
        saved_files = saved.get("files")
    else:
        url = f"{_HF}/api/models/{model_id}/revision/{urllib.parse.quote(revision, safe='')}"
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
                info = json.loads(response.read(_MAX_INDEX_BYTES))
            commit = info["sha"]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            raise AguardenteError(f"não foi possível fixar a revisão de {model_id}: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AguardenteError("Hugging Face não retornou um commit imutável válido")
    return FetchPlan(model_id=model_id, revision=commit, requested_revision=revision,
                     dest=dest, files=tuple(RemoteFile(**file) for file in saved_files)
                     if saved_files is not None else list_files(model_id, commit))


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

    pending = plan.missing()
    plan.dest.mkdir(parents=True, exist_ok=True)
    provenance = plan.dest / ".aguardente-source.json"
    temporary = provenance.with_suffix(".tmp")
    temporary.write_text(json.dumps({"model": plan.model_id, "commit": plan.revision,
                                      "requested_revision": plan.requested_revision,
                                      "files": [asdict(file) for file in plan.files],
                                      "sha256": {f.path: f.sha256 for f in plan.files if f.sha256}}, indent=2))
    temporary.replace(provenance)
    if not pending:
        return plan.dest

    exe = require_aria2()

    lines: list[str] = []
    for f in pending:
        target = safe_join(plan.dest, f.path)
        lines.append(f.url(plan.model_id, plan.revision))
        lines.append(f"  dir={target.parent}")
        lines.append(f"  out={target.name}")
        if f.sha256:
            lines.append(f"  checksum=sha-256={f.sha256}")
    input_file = plan.dest / ".aguardente-fetch.txt"
    input_file.write_text("\n".join(lines) + "\n")
    input_file.chmod(0o600)

    cmd = [
        exe,
        "--input-file", str(input_file),
        "--continue=true",
        "--check-integrity=true",
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

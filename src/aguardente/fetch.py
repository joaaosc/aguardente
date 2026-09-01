"""Download de modelos com aria2c.

Um teacher de 4 B são ~7,5 GB. Numa conexão instável, um download que morre aos
90 % e recomeça do zero é a diferença entre vinte minutos e uma tarde perdida.
O `aria2c` retoma de onde parou e abre várias conexões por arquivo; o servidor
do Hugging Face responde `Accept-Ranges: bytes`, então a retomada funciona de
fato.

Só se baixa o que o pipeline usa. READMEs, licenças e pesos em formatos
alternativos ficam de fora — num repositório com `.bin` **e** `.safetensors`
isso corta metade do tráfego.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .errors import AguardenteError

_HF = "https://huggingface.co"
_TIMEOUT = 30

# Padrões que o carregamento do transformers exige. Ordem sem importância.
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

# Excluídos mesmo quando casam por sufixo: variantes que não usamos.
EXCLUDE_PARTS = ("onnx/", "openvino/", "coreml/", "gguf/", "/consolidated")


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size: int

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
        """Arquivos ausentes ou de tamanho errado — um download truncado conta."""
        out = []
        for f in self.files:
            local = self.dest / f.path
            if not local.is_file() or (f.size and local.stat().st_size != f.size):
                out.append(f)
        return tuple(out)

    @property
    def pending_bytes(self) -> int:
        return sum(f.size for f in self.missing())


def _wanted(path: str) -> bool:
    if any(part in path for part in EXCLUDE_PARTS):
        return False
    name = path.rsplit("/", 1)[-1]
    return name in WANTED_EXACT or path.endswith(WANTED_SUFFIX)


def list_files(model_id: str, revision: str = "main") -> tuple[RemoteFile, ...]:
    """Lista os arquivos do repositório com tamanho, sem baixar nada."""
    url = f"{_HF}/api/models/{model_id}/tree/{revision}?recursive=1"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            entries = json.load(r)
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

    files = []
    for e in entries:
        if e.get("type") != "file" or not _wanted(e["path"]):
            continue
        size = e.get("size") or (e.get("lfs") or {}).get("size") or 0
        files.append(RemoteFile(path=e["path"], size=int(size)))

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
    """Baixa o que falta. Idempotente — o que já está completo é pulado."""
    exe = require_aria2()
    pending = plan.missing()
    if not pending:
        return plan.dest

    plan.dest.mkdir(parents=True, exist_ok=True)

    # Formato do arquivo de entrada do aria2: URL numa linha, opções indentadas.
    lines: list[str] = []
    for f in pending:
        target = plan.dest / f.path
        lines.append(f.url(plan.model_id, plan.revision))
        lines.append(f"  dir={target.parent}")
        lines.append(f"  out={target.name}")
    input_file = plan.dest / ".aguardente-fetch.txt"
    input_file.write_text("\n".join(lines) + "\n")

    cmd = [
        exe,
        "--input-file", str(input_file),
        "--continue=true",                      # retoma de onde parou
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--max-concurrent-downloads={concurrent}",
        "--min-split-size=8M",
        f"--max-tries={max_tries}",
        f"--retry-wait={retry_wait}",
        "--timeout=60",
        "--connect-timeout=30",
        "--auto-file-renaming=false",           # retomar, não criar file.1
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
            hint="O download é retomável: rode o mesmo comando de novo e ele "
                 "continua de onde parou.",
        )

    still_missing = plan.missing()
    if still_missing:
        names = ", ".join(f.path for f in still_missing[:3])
        raise AguardenteError(
            f"{len(still_missing)} arquivo(s) incompletos após o download: {names}",
            hint="Rode de novo — o aria2c retoma o que falta.",
        )
    return plan.dest


def fetch_model(
    model_id: str, dest: str | Path, *, revision: str = "main",
    on_line: Callable[[str], None] | None = None, **kw: Any,
) -> FetchPlan:
    """Conveniência: planeja e baixa numa chamada."""
    plan = plan_fetch(model_id, dest, revision=revision)
    fetch(plan, on_line=on_line, **kw)
    return plan

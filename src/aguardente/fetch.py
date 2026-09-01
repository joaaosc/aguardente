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

# Uma resposta de índice legítima tem alguns KB. O limite existe para que uma
# resposta hostil ou defeituosa não consuma memória sem fim antes de falhar.
_MAX_INDEX_BYTES = 8 * 1024 * 1024

# Formato dos identificadores do Hugging Face: `nome` ou `namespace/nome`.
# Sem esta validação, um identificador como `../../api/interno` ou
# `a/b?x=1#frag` reescreve a URL e aponta a requisição para outro lugar.
_MODEL_ID = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$")


def validate_model_id(model_id: str) -> str:
    """Recusa identificadores capazes de manipular a URL da API."""
    if not isinstance(model_id, str) or not _MODEL_ID.match(model_id):
        raise AguardenteError(
            f"identificador de modelo inválido: {model_id!r}",
            hint="Use o formato `namespace/nome`, como `Qwen/Qwen3-4B`.",
        )
    return model_id

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


def safe_join(base: Path, relative: str) -> Path:
    """Junta um caminho vindo da rede ao destino, recusando qualquer escape.

    O índice de arquivos de um repositório remoto é escrito por quem o publica.
    Um caminho como `../../.ssh/authorized_keys` ou `/etc/cron.d/x` faria a
    escrita cair fora do destino — e caminhos absolutos são especialmente
    traiçoeiros, porque `Path("destino") / "/etc/x"` descarta o destino inteiro
    em vez de concatenar.

    Recusa: caminhos absolutos, componentes `..`, separadores do Windows, e
    qualquer resultado que, depois de resolvido, não esteja sob `base`.
    """
    if not relative or relative.strip() != relative:
        raise AguardenteError(f"caminho remoto inválido: {relative!r}")

    if "\\" in relative or ":" in relative:
        raise AguardenteError(
            f"caminho remoto com separador ou esquema inesperado: {relative!r}",
            hint="O repositório pode estar comprometido.",
        )

    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise AguardenteError(
            f"caminho remoto tenta sair do destino: {relative!r}",
            hint="O repositório pode estar comprometido. Nenhum arquivo foi gravado.",
        )

    base_resolved = base.resolve()
    target = (base_resolved / candidate).resolve()
    # `is_relative_to` compara depois de resolver symlinks — cobre o caso em que
    # um diretório intermediário aponta para fora.
    if not target.is_relative_to(base_resolved):
        raise AguardenteError(
            f"caminho remoto resolve para fora do destino: {relative!r}",
            hint="O repositório pode estar comprometido. Nenhum arquivo foi gravado.",
        )
    return base_resolved / candidate


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """Um arquivo anunciado pelo repositório remoto.

    O `path` é validado na construção: um objeto destes nunca carrega um
    caminho capaz de escapar do destino.
    """

    path: str
    size: int

    def __post_init__(self) -> None:
        # Valida contra uma base sintética — só a forma do caminho importa aqui.
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
        """Arquivos ausentes ou de tamanho errado — um download truncado conta."""
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
    """Se o arquivo interessa ao pipeline — e se o caminho é seguro.

    A checagem de segurança vem primeiro de propósito: um caminho hostil que
    termina em `.safetensors` passaria pelo filtro de extensão sem ela.
    """
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
    """Lista os arquivos do repositório com tamanho, sem baixar nada."""
    validate_model_id(model_id)
    url = f"{_HF}/api/models/{model_id}/tree/{revision}?recursive=1"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            bruto = r.read(_MAX_INDEX_BYTES + 1)
            if len(bruto) > _MAX_INDEX_BYTES:
                raise AguardenteError(
                    f"índice de {model_id} excede {_MAX_INDEX_BYTES // 1024 // 1024} MB",
                    hint="Resposta implausível para um índice de repositório.",
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
            # `_wanted` já deveria ter barrado; se chegou aqui, o repositório
            # está a tentar algo. Ignora a entrada e segue com as demais.
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
    """Baixa o que falta. Idempotente — o que já está completo é pulado."""
    # Valores absurdos fazem o aria2 falhar com mensagens obscuras, ou abrir
    # conexões demais contra o servidor. Limitar aqui dá erro claro e cedo.
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

    # Formato do arquivo de entrada do aria2: URL numa linha, opções indentadas.
    lines: list[str] = []
    for f in pending:
        target = safe_join(plan.dest, f.path)
        lines.append(f.url(plan.model_id, plan.revision))
        lines.append(f"  dir={target.parent}")
        lines.append(f"  out={target.name}")
    input_file = plan.dest / ".aguardente-fetch.txt"
    input_file.write_text("\n".join(lines) + "\n")
    # Lista de URLs e caminhos locais: sem interesse para outros usuários da máquina.
    input_file.chmod(0o600)

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

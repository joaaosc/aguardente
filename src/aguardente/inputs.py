"""Content identities and semantic compatibility of distillation inputs."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .errors import AguardenteError


@lru_cache(maxsize=512)
def _digest(path: str, size: int, mtime: int) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    return _digest(str(p), stat.st_size, stat.st_mtime_ns)


def local_identity(ref: str | None) -> dict | None:
    if not ref:
        return None
    p = Path(ref).expanduser()
    if p.is_file():
        return {p.name: file_identity(p)}
    if p.is_dir():
        return {str(f.relative_to(p)): file_identity(f) for f in sorted(p.rglob("*"))
                if f.is_file() and not {".cache", ".git"}.intersection(f.relative_to(p).parts)
                and f.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}}
    return None


def tokenizer_identity(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise AguardenteError("não foi possível comprovar equivalência dos tokenizers",
                              hint="Destilação de logits requer tokenizers fast com mapeamento e processamento idênticos.")
    config = json.loads(backend.to_str())
    # Runtime batching options do not alter the definition of an individual token.
    config.pop("padding", None)
    config.pop("truncation", None)
    config["special_token_ids"] = {k: getattr(tokenizer, k, None) for k in
                                   ("bos_token_id", "eos_token_id", "unk_token_id")}
    config["chat_template"] = getattr(tokenizer, "chat_template", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def require_same_tokenizer(teacher, student) -> None:
    if tokenizer_identity(teacher) != tokenizer_identity(student):
        raise AguardenteError("tokenizers incompatíveis para destilação de logits",
                              hint="Mesmo vocab_size não basta: tokens, IDs, normalização e regras de segmentação devem coincidir. Use um student da mesma família de tokenizer.")

"""Carregamento e processamento de textos para calibração."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"

# Datasets alternativos para fallback caso o padrão falhe
_FALLBACKS: tuple[tuple[str, str | None], ...] = (
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),
    ("mindchain/wikitext2", None),
)


def _stream_texts(dataset: str, config: str | None, split: str,
                  limit: int, min_chars: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    out: list[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) >= min_chars:
            out.append(text)
            if len(out) >= limit:
                break
    return out


def load_texts(
    *,
    dataset: str | None = None,
    config: str | None = DEFAULT_CONFIG,
    split: str = "train",
    limit: int = 256,
    min_chars: int = 200,
) -> list[str]:
    """Carrega amostras de texto não vazias para calibração."""
    from .errors import AguardenteError

    attempts = [(dataset, config)] if dataset else list(_FALLBACKS)
    errors: list[str] = []
    for name, cfg in attempts:
        try:
            out = _stream_texts(name, cfg, split, limit, min_chars)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if out:
            return out
        errors.append(f"{name}: nenhuma amostra com >= {min_chars} caracteres")

    raise AguardenteError(
        "não foi possível carregar dados de calibração:\n    "
        + "\n    ".join(errors),
        hint="Informe --calib-dataset no formato namespace/nome ou utilize --calib-file com um arquivo .txt local.",
    )


def load_texts_from_file(path: str, *, limit: int = 256, min_chars: int = 200) -> list[str]:
    """Carrega amostras de calibração a partir de um arquivo de texto local."""
    from pathlib import Path

    from .errors import AguardenteError

    p = Path(path).expanduser()
    if not p.is_file():
        raise AguardenteError(
            f"arquivo de calibração não encontrado: {path}",
            hint="Verifique o caminho informado em --calib-file.",
        )

    raw = p.read_text(encoding="utf-8", errors="replace")
    out = [par.strip() for par in raw.split("\n\n") if len(par.strip()) >= min_chars][:limit]
    if not out:
        # Fallback: agrupa linhas contínuas quando o arquivo não usar quebras duplas
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        current: list[str] = []
        current_len = 0
        for line in lines:
            current.append(line)
            current_len += len(line)
            if current_len >= min_chars:
                out.append(" ".join(current))
                current = []
                current_len = 0
                if len(out) >= limit:
                    break
        # O resto de `current` fica sempre abaixo de `min_chars` — o laço fecha um
        # bloco assim que o limiar é atingido — então não há bloco final a emitir.
    if not out:
        raise AguardenteError(
            f"{path} não contém parágrafos com >= {min_chars} caracteres",
            hint="Separe os blocos de texto por linhas em branco ou forneça textos mais longos.",
        )
    return out


def make_batches(
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int = 2,
    seq_len: int = 512,
    device: str | None = None,
) -> Iterator[dict[str, "torch.Tensor"]]:
    """Tokeniza e agrupa as amostras em batches de tamanho fixo."""
    import torch

    ensure_pad_token(tokenizer)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        if len(chunk) < batch_size:
            break
        enc = tokenizer(
            chunk, return_tensors="pt", padding="max_length",
            truncation=True, max_length=seq_len,
        )
        batch = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        if device:
            batch = {k: v.to(device) for k, v in batch.items()}
        yield batch


def ensure_pad_token(tokenizer: Any) -> Any:
    """Configura o token de preenchimento (pad) caso não esteja definido no tokenizer."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

"""Dados de calibração.

Usados para pontuar importância antes de podar. Entradas **reais** importam:
ruído aleatório não exercita a estrutura aprendida dos pesos, e a pontuação
resultante não distingue nada.

Poucos lotes bastam — o sinal é grosseiro por natureza.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import torch

# O `huggingface_hub` recente exige identificadores no formato `namespace/name`;
# o antigo "wikitext" solto falha com HfUriError. O canónico agora é este.
DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"

# Tentados em ordem quando o principal falha — versões diferentes de `datasets`
# e espelhos movidos são comuns o bastante para justificar o fallback.
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
    """Amostras de texto não vazias. Linhas curtas do WikiText são cabeçalhos.

    Entradas **reais** importam: ruído aleatório não exercita a estrutura
    aprendida dos pesos, e a pontuação de importância resultante não distingue
    nada.
    """
    from .errors import AguardenteError

    attempts = [(dataset, config)] if dataset else list(_FALLBACKS)
    errors: list[str] = []
    for name, cfg in attempts:
        try:
            out = _stream_texts(name, cfg, split, limit, min_chars)
        except Exception as e:  # noqa: BLE001 — qualquer falha justifica o próximo
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if out:
            return out
        errors.append(f"{name}: nenhuma amostra com >= {min_chars} caracteres")

    raise AguardenteError(
        "não foi possível carregar dados de calibração:\n    "
        + "\n    ".join(errors),
        hint="Passe --calib-dataset com um identificador no formato namespace/name, "
             "ou use --calib-file com um .txt local.",
    )


def load_texts_from_file(path: str, *, limit: int = 256, min_chars: int = 200) -> list[str]:
    """Alternativa offline: um arquivo de texto, dividido em parágrafos."""
    from pathlib import Path

    from .errors import AguardenteError

    raw = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
    out = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= min_chars][:limit]
    if not out:
        raise AguardenteError(
            f"{path} não tem parágrafos com >= {min_chars} caracteres",
            hint="Separe os trechos por linha em branco.",
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
    """Tokeniza e agrupa em lotes de shape fixo."""
    import torch

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
    """Muitos tokenizers de LLM causal não definem pad; sem isso o batching falha."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

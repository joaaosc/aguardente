"""Carregamento e processamento de textos para calibração."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator
import random

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"

# Datasets alternativos para fallback caso o padrão falhe
_FALLBACKS: tuple[tuple[str, str | None], ...] = (
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),
)


def _stream_texts(dataset: str, config: str | None, split: str,
                  limit: int, min_chars: int, seed: int | None = None,
                  text_column: str = "text") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=4096)
    out: list[str] = []
    for row in ds:
        text = row.get(text_column)
        if not isinstance(text, str):
            raise ValueError(f"dataset sem coluna textual {text_column!r}")
        text = text.strip()
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
    seed: int | None = None,
    text_column: str = "text",
) -> list[str]:
    """Carrega amostras de texto não vazias para calibração."""
    from .errors import AguardenteError

    attempts = [(dataset, config)] if dataset else list(_FALLBACKS)
    errors: list[str] = []
    for name, cfg in attempts:
        try:
            out = _stream_texts(name, cfg, split, limit, min_chars, seed, text_column)
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


def load_texts_from_file(path: str, *, limit: int = 256, min_chars: int = 200,
                         seed: int | None = None, tokenizer: Any = None) -> list[str]:
    """Carrega amostras de calibração a partir de um arquivo de texto local."""
    from pathlib import Path

    from .errors import AguardenteError

    p = Path(path).expanduser()
    if not p.is_file():
        raise AguardenteError(
            f"arquivo de calibração não encontrado: {path}",
            hint="Verifique o caminho informado em --calib-file.",
        )

    raw = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        import json
        paragraphs = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AguardenteError(f"JSONL inválido em {path}:{number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise AguardenteError(f"JSONL exige um objeto por linha em {path}:{number}")
            if isinstance(row.get("text"), str):
                paragraphs.append(row["text"])
            elif "messages" in row and tokenizer is not None and tokenizer.chat_template:
                paragraphs.append(tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                                               add_generation_prompt=False,
                                                               **row.get("template_kwargs", {})))
            else:
                raise AguardenteError("JSONL exige text ou messages com chat_template do tokenizer")
    else:
        paragraphs = raw.split("\n\n")
    if seed is not None:
        random.Random(seed).shuffle(paragraphs)
    out = [par.strip() for par in paragraphs if len(par.strip()) >= min_chars][:limit]
    if not out and p.suffix != ".jsonl":
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


def make_packed_batches(tokenizer: Any, texts: list[str], *, batch_size: int = 2,
                        seq_len: int = 512, max_samples: int | None = None,
                        device: str | None = None,
                        loss_masks: dict[str, list[int]] | None = None) -> Iterator[dict[str, "torch.Tensor"]]:
    """Pack complete documents with EOS boundaries; never truncate long documents.

    One token overlaps adjacent windows, so a boundary does not lose a causal
    target. Only the final window is padded, and its mask retains real EOS.
    """
    import torch
    if batch_size < 1 or seq_len < 2:
        raise ValueError("batch_size >= 1 and seq_len >= 2 required")
    ensure_pad_token(tokenizer)
    eos = tokenizer.eos_token_id
    if eos is None or tokenizer.pad_token_id is None:
        raise ValueError("packing requires a trained EOS and a padding token")
    buffer: list[int] = []
    rows: list[list[int]] = []
    supervision: list[int] = []
    row_masks: list[list[int]] = []
    emitted = 0

    def emit():
        ids = torch.full((len(rows), seq_len), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        targets = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
            targets[i, :len(row)] = torch.tensor(row_masks[i])
        batch = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
        if loss_masks is not None:
            batch["loss_mask"] = targets.to(device)
        return batch

    for text in texts:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not tokens:
            continue
        target_mask = list(loss_masks[text]) if loss_masks is not None else [1] * len(tokens)
        if len(target_mask) != len(tokens) or any(x not in (0, 1) for x in target_mask):
            raise ValueError("loss mask must match document tokenization")
        buffer.extend(tokens)
        supervision.extend(target_mask)
        if buffer[-1] != eos:
            buffer.append(eos)
            supervision.append(target_mask[-1])
        offset = 0
        while len(buffer) - offset >= seq_len:
            row = buffer[offset:offset + seq_len]
            targets = supervision[offset:offset + seq_len]
            offset += seq_len - 1
            if not any(targets[1:]):
                continue
            rows.append(row)
            row_masks.append(targets)
            emitted += 1
            if len(rows) == batch_size or emitted == max_samples:
                yield emit()
                rows = []
                row_masks = []
            if emitted == max_samples:
                return
        buffer = buffer[offset:]
        supervision = supervision[offset:]
    if len(buffer) >= 2 and any(supervision[1:]):
        rows.append(buffer)
        row_masks.append(supervision)
    if rows:
        yield emit()

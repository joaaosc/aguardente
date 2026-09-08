"""Avaliação de qualidade e métricas de perplexidade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True, slots=True)
class Perplexity:
    """Resultado da medição de perplexidade."""

    value: float
    tokens: int
    windows: int

    def __str__(self) -> str:
        return f"{self.value:.2f} ({self.tokens:,} tokens)"


def perplexity(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_length: int | None = None,
    stride: int = 512,
    device: str | None = None,
) -> Perplexity:
    """Calcula a perplexidade do modelo em um texto utilizando janela deslizante."""
    import torch

    dev = device or next(model.parameters()).device
    max_len = max_length or int(getattr(model.config, "max_position_embeddings", 2048))
    max_len = min(max_len, 4096)

    ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
    n = ids.size(1)
    if n < 2:
        raise ValueError("texto curto demais para medir perplexidade")
    if max_len < 2 or stride < 1 or stride >= max_len:
        raise ValueError("perplexity requires 1 <= stride < max_length to score every causal target")

    nll_sum = torch.zeros((), dtype=torch.float64)
    counted = 0
    windows = 0
    prev_end = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for begin in range(0, n, stride):
                end = min(begin + max_len, n)
                target_len = end - prev_end
                if target_len <= 0:
                    continue

                window = ids[:, begin:end]
                targets = window.clone()
                targets[:, :-target_len] = -100

                valid = int((targets[:, 1:] != -100).sum().item())
                if valid <= 0:
                    continue
                out = model(input_ids=window, labels=targets)
                nll_sum += out.loss.detach().cpu().double() * valid
                counted += valid
                windows += 1

                prev_end = end
                if end == n:
                    break
    finally:
        model.train(was_training)

    return Perplexity(value=float(torch.exp(nll_sum / counted)),
                      tokens=counted, windows=windows)


def response_perplexity(model, tokenizer, texts, loss_masks, *, max_length=512, device=None):
    """Score only supplied assistant targets; padding and prompt tokens stay excluded."""
    import math
    import torch
    from .calibration import make_packed_batches
    dev = device or next(model.parameters()).device
    total = 0.0
    tokens = windows = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in make_packed_batches(tokenizer, texts, seq_len=max_length,
                    batch_size=1, device=dev, loss_masks=loss_masks):
                active = batch["attention_mask"][:, 1:].bool() & batch["attention_mask"][:, :-1].bool() & batch["loss_mask"][:, 1:].bool()
                target = batch["input_ids"][:, 1:].masked_fill(~active, -100)
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
                total += torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                    target.reshape(-1), ignore_index=-100, reduction="sum").item()
                tokens += int(active.sum())
                windows += 1
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("no assistant targets for evaluation")
    return Perplexity(math.exp(total / tokens), tokens, windows)


def perplexity_on_wikitext(
    model: Any, tokenizer: Any, *, max_chars: int = 20_000, split: str = "test", **kw: Any
) -> Perplexity:
    """Calcula a perplexidade sobre o conjunto de teste do WikiText."""
    from .calibration import load_texts

    texts = load_texts(split=split, limit=512, min_chars=1)
    joined = "\n\n".join(texts)[:max_chars]
    kw.setdefault("max_length", 512)
    kw.setdefault("stride", 256)
    return perplexity(model, tokenizer, joined, **kw)


def recovery_fraction(teacher: float, pruned: float, recovered: float) -> float:
    """Calcula a fração da perda de desempenho recuperada após a destilação."""
    drop = pruned - teacher
    if drop <= 0:
        return 1.0
    return (pruned - recovered) / drop

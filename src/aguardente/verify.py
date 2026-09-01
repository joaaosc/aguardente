"""Medição de qualidade.

Perplexidade responde "o modelo ainda presta?". É diferente de PSNR, que
responde "a conversão preservou a numérica?" — uma conversão perfeita de um
modelo mal recuperado dá PSNR alto e perplexidade péssima.

O `coreai.llm.eval` da Apple é um stub (`parser.error("coming soon")`), então
esta parte é nossa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True, slots=True)
class Perplexity:
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
    """Perplexidade por janela deslizante.

    A janela deslizante existe porque avaliar em blocos disjuntos penaliza
    injustamente os primeiros tokens de cada bloco — eles não têm contexto.
    Aqui só os `stride` últimos tokens de cada janela contam para a perda; o
    resto serve de contexto e é mascarado com -100.
    """
    import torch

    dev = device or next(model.parameters()).device
    max_len = max_length or int(getattr(model.config, "max_position_embeddings", 2048))
    max_len = min(max_len, 4096)  # janelas gigantes não melhoram a medida e custam RAM

    ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
    n = ids.size(1)
    if n < 2:
        raise ValueError("texto curto demais para medir perplexidade")

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
                target_len = end - prev_end       # só o trecho novo conta
                if target_len <= 0:
                    continue

                window = ids[:, begin:end]
                targets = window.clone()
                targets[:, :-target_len] = -100

                out = model(input_ids=window, labels=targets)
                # A perda do HF é a média sobre os alvos não mascarados; o shift
                # de labels já é feito internamente, por isso são (target_len-1).
                valid = max(1, target_len - 1)
                nll_sum += out.loss.detach().double().cpu() * valid
                counted += valid
                windows += 1

                prev_end = end
                if end == n:
                    break
    finally:
        model.train(was_training)

    return Perplexity(value=float(torch.exp(nll_sum / counted)),
                      tokens=counted, windows=windows)


def perplexity_on_wikitext(
    model: Any, tokenizer: Any, *, max_chars: int = 200_000, **kw: Any
) -> Perplexity:
    """Perplexidade no WikiText-2 — a referência que os READMEs da Apple usam."""
    from .calibration import load_texts

    texts = load_texts(split="test", limit=512, min_chars=1)
    joined = "\n\n".join(texts)[:max_chars]
    return perplexity(model, tokenizer, joined, **kw)


def recovery_fraction(teacher: float, pruned: float, recovered: float) -> float:
    """Que fração da queda causada pela poda foi recuperada.

    1.0 = voltou ao nível do teacher; 0.0 = a recuperação não mudou nada;
    negativo = a recuperação piorou o modelo.
    """
    drop = pruned - teacher
    if drop <= 0:
        return 1.0
    return (pruned - recovered) / drop

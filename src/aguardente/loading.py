"""Carregamento de modelos e tokenizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import AguardenteError


def pick_device(prefer: str | None = None) -> str:
    """Seleciona o dispositivo de execução (MPS quando disponível, ou CPU)."""
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> Any:
    """Seleciona o tipo de precisão para o dispositivo (float16 para MPS, float32 para CPU)."""
    import torch

    return torch.float16 if device == "mps" else torch.float32


def load_causal_lm(ref: str, *, device: str | None = None, dtype: Any = None) -> tuple[Any, Any]:
    """Carrega o modelo causal e tokenizer a partir de um identificador do Hub ou diretório local."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .calibration import ensure_pad_token

    dev = pick_device(device)
    dt = dtype if dtype is not None else pick_dtype(dev)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            ref, dtype=dt, low_cpu_mem_usage=True,
        )
        tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(ref))
    except Exception as e:  # noqa: BLE001
        raise AguardenteError(
            f"não foi possível carregar {ref}: {e}",
            hint="Para diretórios locais, certifique-se de que config.json, os arquivos .safetensors e o tokenizer estão presentes.",
        ) from e

    model.to(dev)
    model.eval()
    return model, tokenizer


def save_pruned(model: Any, tokenizer: Any, out_dir: Path) -> Path:
    """Salva o modelo e o tokenizer no formato transformers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir

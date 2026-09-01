"""Carregamento de modelos, com as escolhas que importam em máquina pequena."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import AguardenteError


def pick_device(prefer: str | None = None) -> str:
    """MPS quando existe. Em Apple Silicon, `cuda` nunca é opção."""
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> Any:
    """float16, não bfloat16: o suporte a bf16 em MPS é irregular."""
    import torch

    return torch.float16 if device == "mps" else torch.float32


def load_causal_lm(ref: str, *, device: str | None = None, dtype: Any = None) -> tuple[Any, Any]:
    """Carrega modelo e tokenizer de um identificador do Hugging Face ou diretório."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .calibration import ensure_pad_token

    dev = pick_device(device)
    dt = dtype if dtype is not None else pick_dtype(dev)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            ref, dtype=dt, low_cpu_mem_usage=True,
        )
        tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(ref))
    except Exception as e:  # noqa: BLE001 — a mensagem do transformers é o que importa
        raise AguardenteError(
            f"não foi possível carregar {ref}: {e}",
            hint="Para um diretório local, confira config.json, os .safetensors e o tokenizer.",
        ) from e

    model.to(dev)
    model.eval()
    return model, tokenizer


def save_pruned(model: Any, tokenizer: Any, out_dir: Path) -> Path:
    """Grava no formato transformers — o que o `coreai.llm.export` consome."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir

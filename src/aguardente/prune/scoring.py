"""Cálculo de importância para poda estruturada."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any, Iterable

from .surgery import _find_layers

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(slots=True)
class Scores:
    """Métricas de importância por componente."""

    ffn: "torch.Tensor"        # [intermediate_size]
    kv_groups: "torch.Tensor"  # [num_key_value_heads]
    layers: "torch.Tensor"     # [num_hidden_layers]

    def top_ffn(self, k: int) -> "torch.Tensor":
        """Retorna os índices ordenados dos k neurônios intermediários mais importantes."""
        return self.ffn.topk(k).indices.sort().values

    def top_kv_groups(self, k: int) -> list[int]:
        """Retorna os índices dos k grupos KV mais importantes."""
        return sorted(self.kv_groups.topk(k).indices.tolist())

    def keep_layers(self, k: int, *, protect: set[int] | None = None) -> list[int]:
        """Retorna os índices das k camadas a manter, preservando as camadas protegidas."""
        import torch

        n = len(self.layers)
        protect = protect if protect is not None else {0, n - 1}
        protect = {i for i in protect if 0 <= i < n}
        if k <= len(protect):
            return sorted(protect)[:k]

        scores = self.layers.clone()
        scores[torch.tensor(sorted(protect), dtype=torch.long)] = float("inf")
        return sorted(scores.topk(k).indices.tolist())


def score_model(
    model: Any,
    batches: Iterable[dict[str, Any]],
    *,
    max_batches: int = 16,
) -> Scores:
    """Calcula os scores de importância a partir de ativações em lotes de calibração."""
    import torch

    layers = _find_layers(model)
    n_layers = len(layers)

    ffn_acc: torch.Tensor | None = None
    kv_acc: torch.Tensor | None = None
    influence = torch.zeros(n_layers, dtype=torch.float64)
    seen = 0
    handles: list[Any] = []

    def _hook_down_proj(_m: Any, inputs: tuple, _out: Any) -> None:
        nonlocal ffn_acc
        x = inputs[0].detach().float()
        v = x.reshape(-1, x.shape[-1]).pow(2).mean(0).sqrt().cpu()
        ffn_acc = v if ffn_acc is None else ffn_acc + v

    def _hook_v_proj(_m: Any, _inputs: tuple, out: Any) -> None:
        nonlocal kv_acc
        y = out.detach().float()
        v = y.reshape(-1, y.shape[-1]).pow(2).mean(0).sqrt().cpu()
        kv_acc = v if kv_acc is None else kv_acc + v

    def _make_block_hook(idx: int):
        def hook(_m: Any, inputs: tuple, out: Any) -> None:
            x = inputs[0]
            y = out[0] if isinstance(out, tuple) else out
            if x is None or y is None or x.shape != y.shape:
                return
            a = x.detach().float().reshape(-1, x.shape[-1])
            b = y.detach().float().reshape(-1, y.shape[-1])
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
            influence[idx] += float(1.0 - cos)
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_block_hook(i)))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            handles.append(layer.mlp.down_proj.register_forward_hook(_hook_down_proj))
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
            handles.append(layer.self_attn.v_proj.register_forward_hook(_hook_v_proj))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in islice(batches, max_batches):
                model(**batch)
                seen += 1
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    if seen == 0 or ffn_acc is None or kv_acc is None:
        raise ValueError("nenhum lote de calibração foi processado")

    cfg = model.config
    n_kv = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    head_dim = kv_acc.numel() // n_kv

    return Scores(
        ffn=ffn_acc / seen,
        kv_groups=(kv_acc / seen).reshape(n_kv, head_dim).mean(dim=1),
        layers=(influence / seen).float(),
    )

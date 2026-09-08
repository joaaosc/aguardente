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

    ffn: "torch.Tensor"        # [num_hidden_layers, intermediate_size]
    kv_groups: "torch.Tensor"  # [num_hidden_layers, num_key_value_heads]
    layers: "torch.Tensor"     # [num_hidden_layers]

    def top_ffn(self, k: int) -> "torch.Tensor":
        """Retorna os índices ordenados dos k neurônios intermediários mais importantes."""
        return self.ffn.topk(k, dim=-1).indices.sort(dim=-1).values

    def top_kv_groups(self, k: int) -> list[int]:
        """Retorna os índices dos k grupos KV mais importantes."""
        return self.kv_groups.topk(k, dim=-1).indices.sort(dim=-1).values.tolist()

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

    ffn_acc: dict[int, torch.Tensor] = {}
    kv_acc: dict[int, torch.Tensor] = {}
    influence = torch.zeros(n_layers, dtype=torch.float64)
    seen = 0
    handles: list[Any] = []
    active_mask = None

    def valid_rows(x):
        flat = x.detach().float().reshape(-1, x.shape[-1])
        return flat[active_mask.reshape(-1).bool()] if active_mask is not None else flat

    def _hook_down_proj(idx, _m: Any, inputs: tuple, _out: Any) -> None:
        x = valid_rows(inputs[0])
        # Output-weight norm accounts for the actual residual contribution.
        v = (x.pow(2).sum(0) * _m.weight.detach().float().pow(2).sum(0)).cpu()
        ffn_acc[idx] = ffn_acc.get(idx, 0) + v

    def _hook_v_proj(idx, _m: Any, _inputs: tuple, out: Any) -> None:
        v = valid_rows(out).pow(2).sum(0).cpu()
        kv_acc[idx] = kv_acc.get(idx, 0) + v

    def _make_block_hook(idx: int):
        def hook(_m: Any, inputs: tuple, out: Any) -> None:
            x = inputs[0]
            y = out[0] if isinstance(out, tuple) else out
            if x is None or y is None or x.shape != y.shape:
                return
            a, b = valid_rows(x), valid_rows(y)
            influence[idx] += float((1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).sum())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_block_hook(i)))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            from functools import partial
            handles.append(layer.mlp.down_proj.register_forward_hook(partial(_hook_down_proj, i)))
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
            from functools import partial
            handles.append(layer.self_attn.v_proj.register_forward_hook(partial(_hook_v_proj, i)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in islice(batches, max_batches):
                active_mask = batch.get("attention_mask")
                if active_mask is not None and not active_mask.any():
                    continue
                model(**batch)
                seen += int(active_mask.sum()) if active_mask is not None else batch["input_ids"].numel()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    if seen == 0 or len(ffn_acc) != n_layers or len(kv_acc) != n_layers:
        raise ValueError("nenhum lote de calibração foi processado")

    cfg = model.config
    n_kv = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    ffn = torch.stack([ffn_acc[i] for i in range(n_layers)]) / seen
    kv = torch.stack([kv_acc[i] for i in range(n_layers)]) / seen
    head_dim = kv.shape[-1] // n_kv

    return Scores(
        ffn=ffn.sqrt(),
        kv_groups=kv.reshape(n_layers, n_kv, head_dim).mean(dim=-1).sqrt(),
        layers=(influence / seen).float(),
    )

"""Aplicação de poda estruturada nos tensores do modelo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from ..arch import Arch
from ..errors import UnsupportedArchitecture
from ..plan import PrunePlan

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn


@dataclass(frozen=True, slots=True)
class PruneReport:
    params_before: int
    params_after: int
    kept_layers: tuple[int, ...]
    kept_kv_groups: tuple[int | tuple[int, ...], ...]
    kept_ffn_count: int

    @property
    def ratio(self) -> float:
        return self.params_before / max(1, self.params_after)


def _slice_linear_out(linear: "nn.Linear", keep: "torch.Tensor") -> "nn.Linear":
    """Mantém apenas as linhas selecionadas na matriz de pesos (reduz out_features)."""
    import torch
    from torch import nn

    new = nn.Linear(linear.in_features, len(keep), bias=linear.bias is not None,
                    device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(linear.weight[keep])
        if linear.bias is not None:
            new.bias.copy_(linear.bias[keep])
    return new


def _slice_linear_in(linear: "nn.Linear", keep: "torch.Tensor") -> "nn.Linear":
    """Mantém apenas as colunas selecionadas na matriz de pesos (reduz in_features)."""
    import torch
    from torch import nn

    new = nn.Linear(len(keep), linear.out_features, bias=linear.bias is not None,
                    device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(linear.weight[:, keep])
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new


def _expand_groups_to_heads(groups: Sequence[int], heads_per_group: int) -> list[int]:
    """Expande os índices de grupos KV para os índices das cabeças de query correspondentes."""
    return [g * heads_per_group + i for g in groups for i in range(heads_per_group)]


def _head_slice_indices(heads: Sequence[int], head_dim: int) -> "torch.Tensor":
    """Converte índices de cabeças em índices contínuos de tensores."""
    import torch
    return torch.tensor([h * head_dim + i for h in heads for i in range(head_dim)],
                        dtype=torch.long)


def _find_layers(model: Any) -> "nn.ModuleList":
    """Localiza o módulo ModuleList contendo as camadas do modelo."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise UnsupportedArchitecture(
        "não foi possível localizar a lista de camadas do modelo",
        hint="Só arquiteturas de LLM causal no formato transformers são suportadas.",
    )


def _prune_mlp(layer: Any, keep: "torch.Tensor") -> None:
    mlp = layer.mlp
    mlp.gate_proj = _slice_linear_out(mlp.gate_proj, keep)
    mlp.up_proj = _slice_linear_out(mlp.up_proj, keep)
    mlp.down_proj = _slice_linear_in(mlp.down_proj, keep)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = len(keep)


def _prune_attention(layer: Any, keep_groups: Sequence[int], src: Arch, dst: Arch) -> None:
    attn = layer.self_attn
    q_heads = _expand_groups_to_heads(keep_groups, src.heads_per_group)

    q_idx = _head_slice_indices(q_heads, src.head_dim)
    kv_idx = _head_slice_indices(keep_groups, src.head_dim)

    attn.q_proj = _slice_linear_out(attn.q_proj, q_idx)
    attn.k_proj = _slice_linear_out(attn.k_proj, kv_idx)
    attn.v_proj = _slice_linear_out(attn.v_proj, kv_idx)
    attn.o_proj = _slice_linear_in(attn.o_proj, q_idx)

    for name, value in (
        ("num_heads", dst.num_attention_heads),
        ("num_attention_heads", dst.num_attention_heads),
        ("num_key_value_heads", dst.num_key_value_heads),
        ("num_key_value_groups", dst.heads_per_group),
        ("hidden_size", dst.hidden_size),
        ("head_dim", dst.head_dim),
    ):
        if hasattr(attn, name):
            setattr(attn, name, value)


def prune_model(
    model: Any,
    plan: PrunePlan,
    *,
    keep_ffn: "torch.Tensor | None" = None,
    keep_groups: Sequence[int] | None = None,
    keep_layers: Sequence[int] | None = None,
) -> PruneReport:
    """Aplica o plano de poda estruturada ao modelo in-place."""
    import torch

    src, dst = plan.source, plan.target
    before = sum(p.numel() for p in model.parameters())
    layers = _find_layers(model)

    if keep_ffn is None:
        keep_ffn = torch.arange(dst.intermediate_size)
    if keep_groups is None:
        keep_groups = list(range(dst.num_key_value_heads))
    if keep_layers is None:
        keep_layers = _default_layer_selection(src.num_hidden_layers, dst.num_hidden_layers)

    keep_ffn = torch.as_tensor(keep_ffn, dtype=torch.long)
    if keep_ffn.shape[-1] != dst.intermediate_size:
        raise ValueError(f"keep_ffn tem {len(keep_ffn)} índices, plano pede {dst.intermediate_size}")
    per_layer_groups = bool(keep_groups) and isinstance(keep_groups[0], (list, tuple))
    if len(keep_groups[0] if per_layer_groups else keep_groups) != dst.num_key_value_heads:
        raise ValueError(f"keep_groups tem {len(keep_groups)}, plano pede {dst.num_key_value_heads}")
    if len(keep_layers) != dst.num_hidden_layers:
        raise ValueError(f"keep_layers tem {len(keep_layers)}, plano pede {dst.num_hidden_layers}")

    for idx in keep_layers:
        layer = layers[idx]
        if dst.intermediate_size != src.intermediate_size:
            _prune_mlp(layer, keep_ffn[idx] if keep_ffn.ndim == 2 else keep_ffn)
        if dst.num_key_value_heads != src.num_key_value_heads:
            _prune_attention(layer, keep_groups[idx] if per_layer_groups else keep_groups, src, dst)

    if dst.num_hidden_layers != src.num_hidden_layers:
        from torch import nn
        kept = nn.ModuleList([layers[i] for i in keep_layers])
        for new_idx, layer in enumerate(kept):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = new_idx
        _replace_layers(model, kept)

    _sync_config(model, dst, keep_layers, src.num_hidden_layers)

    return PruneReport(
        params_before=before,
        params_after=sum(p.numel() for p in model.parameters()),
        kept_layers=tuple(keep_layers),
        kept_kv_groups=(tuple(tuple(row) for row in keep_groups) if per_layer_groups
                        else tuple(keep_groups)),
        kept_ffn_count=keep_ffn.shape[-1],
    )


def _replace_layers(model: Any, new_layers: Any) -> None:
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        parts = path.split(".")
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            getattr(obj, parts[-1])
        except AttributeError:
            continue
        setattr(obj, parts[-1], new_layers)
        return
    raise UnsupportedArchitecture("não foi possível substituir a lista de camadas")


def _default_layer_selection(total: int, keep: int) -> list[int]:
    """Distribui os cortes uniformemente, preservando a primeira e a última camada."""
    if keep >= total:
        return list(range(total))
    if keep <= 2:
        return [0, total - 1][:keep]
    middle = [round(1 + i * (total - 2) / (keep - 1)) for i in range(1, keep - 1)]
    return sorted({0, *middle, total - 1})[:keep]


def _resize_per_layer_lists(cfg: Any, keep_layers: Sequence[int], src_layers: int) -> list[str]:
    """Refatia as listas do config que descrevem uma camada por posição.

    Configs recentes do transformers carregam `layer_types` — e famílias com
    atenção alternada carregam listas equivalentes — com um item por camada, e
    validam o comprimento contra `num_hidden_layers` ao reinstanciar. Sem
    refatiar, o config gravado descreve 28 camadas para um modelo com 17 e o
    checkpoint podado não volta a carregar.
    """
    if src_layers <= 0:
        return []
    ajustados = []
    # Schema fields, never infer semantics from coincidental list length.
    for nome in ("layer_types", "attention_types"):
        valor = getattr(cfg, nome, None)
        if not isinstance(valor, (list, tuple)):
            continue
        if len(valor) != src_layers:
            continue
        refatiado = [valor[i] for i in keep_layers]
        setattr(cfg, nome, type(valor)(refatiado) if isinstance(valor, tuple) else refatiado)
        ajustados.append(nome)
    return ajustados


def _sync_config(model: Any, dst: Arch, keep_layers: Sequence[int] = (),
                 src_layers: int = 0) -> list[str]:
    """Atualiza o objeto de configuração do modelo com as novas dimensões."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return []
    dimensoes = (
        ("intermediate_size", dst.intermediate_size),
        ("num_hidden_layers", dst.num_hidden_layers),
        ("num_attention_heads", dst.num_attention_heads),
        ("num_key_value_heads", dst.num_key_value_heads),
    )
    ajustados = []
    for alvo in (cfg, getattr(cfg, "text_config", None)):
        if alvo is None:
            continue
        # As listas por camada são refatiadas antes das dimensões: alguns
        # configs validam o comprimento no próprio `__setattr__`.
        ajustados += _resize_per_layer_lists(alvo, keep_layers, src_layers)
        for name, value in dimensoes:
            if hasattr(alvo, name):
                setattr(alvo, name, value)
    return ajustados

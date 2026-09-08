"""Ridge reconstruction of residual projections after structured column removal.

For kept activations X, fit a correction to W_kept against the removed output:
  delta = (X.T X + damping * mean(diag(X.T X)) I)^-1 X.T residual
The original teacher is never modified while collecting these sufficient stats.
"""
from dataclasses import dataclass
from itertools import islice

from .surgery import _find_layers, _head_slice_indices, _expand_groups_to_heads
from ..errors import InsufficientResources


@dataclass
class Projection:
    layer: int
    path: str
    keep: object
    weight: object = None


def fit_projections(model, plan, keep_ffn, keep_groups, keep_layers, batches_factory,
                    *, max_batches, damping=0.01, max_stats_bytes=256 * 1024**2):
    import torch
    projections = []
    for i in keep_layers:
        if plan.target.intermediate_size < plan.source.intermediate_size:
            projections.append(Projection(i, "mlp.down_proj", keep_ffn[i].cpu()))
        if plan.target.num_key_value_heads < plan.source.num_key_value_heads:
            heads = _expand_groups_to_heads(keep_groups[i], plan.source.heads_per_group)
            projections.append(Projection(i, "self_attn.o_proj", _head_slice_indices(heads, plan.source.head_dim)))
    layers = _find_layers(model)
    groups, group, used = [], [], 0
    for item in projections:
        module = layers[item.layer].get_submodule(item.path)
        k = len(item.keep)
        required = 4 * (k*k + k*module.out_features)
        if required > max_stats_bytes:
            raise InsufficientResources("a reconstrução desta projeção excede o orçamento de estatísticas",
                                        hint="Use uma máquina maior ou --no-reconstruction; a destilação global continua disponível.")
        if group and used + required > max_stats_bytes:
            groups.append(group)
            group, used = [], 0
        group.append(item)
        used += required
    if group:
        groups.append(group)
    was_training = model.training
    model.eval()
    try:
        for group in groups:
            stats, handles = {}, []
            mask = None
            for item in group:
                module = layers[item.layer].get_submodule(item.path)
                k, h = len(item.keep), module.out_features
                stats[id(item)] = [torch.zeros(k, k), torch.zeros(k, h), 0]

                def hook(module, inputs, output, item=item):
                    x = inputs[0].detach().float().reshape(-1, module.in_features)
                    y = output.detach().float().reshape(-1, module.out_features)
                    if mask is not None:
                        valid = mask.reshape(-1).bool()
                        x, y = x[valid], y[valid]
                    if not len(x):
                        return
                    keep = item.keep.to(x.device)
                    selected = x[:, keep]
                    base = module.weight.detach().float()[:, keep]
                    residual = y - selected @ base.T
                    if module.bias is not None:
                        residual -= module.bias.detach().float()
                    gram, rhs, count = stats[id(item)]
                    gram.add_((selected.T @ selected).cpu())
                    rhs.add_((selected.T @ residual).cpu())
                    stats[id(item)][2] = count + len(x)
                handles.append(module.register_forward_hook(hook))
            try:
                with torch.no_grad():
                    for batch in islice(batches_factory(), max_batches):
                        mask = batch.get("attention_mask")
                        model(**batch, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            for item in group:
                gram, rhs, count = stats.pop(id(item))
                if not count:
                    raise ValueError("reconstruction received no valid calibration tokens")
                gram, rhs = gram.double(), rhs.double()
                ridge = max(float(gram.diagonal().mean()) * damping, 1e-8)
                gram.diagonal().add_(ridge)
                delta = torch.linalg.solve(gram, rhs).T.float()
                module = layers[item.layer].get_submodule(item.path)
                base = module.weight.detach().float().cpu()[:, item.keep]
                item.weight = base + delta
                if not torch.isfinite(item.weight).all():
                    raise ValueError("non-finite reconstruction weights")
    finally:
        model.train(was_training)
    return projections


def apply_projections(model, projections, keep_layers):
    """Apply fits to the already-pruned model; return original slices for comparison."""
    import torch
    layers = _find_layers(model)
    originals = []
    with torch.no_grad():
        for item in projections:
            module = layers[keep_layers.index(item.layer)].get_submodule(item.path)
            originals.append((module, module.weight.detach().cpu().clone()))
            module.weight.copy_(item.weight.to(module.weight))
    return originals

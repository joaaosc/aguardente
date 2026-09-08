"""Adaptação de baixo posto (LoRA) para camadas lineares."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..errors import AguardenteError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


__all__ = ["LoRAError", "LoRALinear", "attach_lora", "merge_lora"]


class LoRAError(AguardenteError, ValueError):
    """Erro em validação, anexação ou fusão de adaptadores LoRA."""


class LoRALinear(torch.nn.Module if TORCH_AVAILABLE else object):  # type: ignore
    """Camada de adaptação de baixo posto (LoRA) para nn.Linear."""

    def __init__(
        self,
        base_layer: Any,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise LoRAError("PyTorch é necessário para LoRALinear")
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A e B são sempre alocados em FP32 no mesmo dispositivo da camada base
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty((rank, in_features), dtype=torch.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((out_features, rank), dtype=torch.float32, device=device)
        )
        # Inicializa A com RNG do torch e B com zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> Any:
        return self.base_layer.weight

    @property
    def bias(self) -> Any:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        original = [(p.detach(), None if p.grad is None else p.grad.detach())
                    for p in (self.lora_A, self.lora_B)]
        res = super()._apply(fn, recurse=recurse)
        if TORCH_AVAILABLE:
            for parameter, (value, grad) in zip((self.lora_A, self.lora_B), original):
                parameter.data = value.to(device=parameter.device, dtype=torch.float32)
                if grad is not None:
                    parameter.grad = grad.to(device=parameter.device, dtype=torch.float32)
        return res

    def forward(self, x: Any) -> Any:
        # Executa a camada base congelada em seu dtype nativo
        base_out = self.base_layer(x)
        # Calcula o delta de baixo posto em FP32
        x_fp32 = x.to(torch.float32)
        delta_fp32 = F.linear(
            F.linear(x_fp32, self.lora_A),
            self.lora_B,
        ) * self.scaling
        # Converte o delta para o dtype de saída da base e preserva formato e viés
        return base_out + delta_fp32.to(base_out.dtype)


def _get_parent_and_child(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, nn.Sequential)) and p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_module(root: Any, name: str, new_mod: Any) -> None:
    parent, child_name = _get_parent_and_child(root, name)
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = new_mod
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = new_mod
    else:
        setattr(parent, child_name, new_mod)


def _matches_target(name: str, target: str) -> bool:
    if name == target:
        return True
    clean_target = target.lstrip(".")
    if name == clean_target:
        return True
    if name.endswith("." + clean_target):
        return True
    return False


def _find_tied_weight_modules(model: Any) -> set[str]:
    """Conservatively protect shared storage, including aliased submodules."""
    owners: dict[tuple, set[str]] = {}
    weighted = {}
    for name, module in model.named_modules(remove_duplicate=False):
        for parameter in module.parameters(recurse=False):
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            owners.setdefault(key, set()).add(name)
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            weighted[name] = (str(weight.device), weight.untyped_storage().data_ptr())
    return {name for name, key in weighted.items() if len(owners.get(key, ())) > 1}


def attach_lora(
    model: Any,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    targets: Sequence[str] = (),
) -> dict[str, Any]:
    """Anexa adaptadores LoRA em camadas nn.Linear do modelo.

    Congela os parâmetros do modelo base e insere matrizes de baixo posto A e B em FP32.
    Se `targets` for vazio, adapta todas as nn.Linear elegíveis, exceto a saída
    de embeddings (`model.get_output_embeddings()`) e tensores amarrados (tied weights).
    Se `targets` não for vazio, valida que cada alvo corresponde a nomes exatos
    ou sufixos de submódulos antes de qualquer congelamento ou mutação.
    """
    if not TORCH_AVAILABLE:
        raise LoRAError("PyTorch é necessário para attach_lora")

    # 1. Validação de rank
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise LoRAError(f"rank deve ser um inteiro positivo, recebido {rank!r}")

    # 2. Validação de alpha
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise LoRAError(f"alpha deve ser finito e positivo, recebido {alpha!r}")

    # 3. Validação de targets
    if isinstance(targets, str):
        targets = (targets,)
    elif not hasattr(targets, "__iter__"):
        raise LoRAError(f"targets deve ser uma sequência de strings, recebido {type(targets).__name__}")
    targets = tuple(targets)
    for t in targets:
        if not isinstance(t, str) or not t.strip():
            raise LoRAError(f"alvo de adaptação deve ser uma string não vazia, recebido {t!r}")

    # 4. Verifica se o modelo já possui adaptadores LoRA
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            raise LoRAError(f"o modelo já possui adaptadores LoRA anexados (módulo {name!r})")

    # 5. Coleta todas as camadas nn.Linear elegíveis (com nome não vazio)
    eligible_linears: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if name and isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
            eligible_linears[name] = mod

    # 6. Seleção de módulos a adaptar
    selected_names: list[str] = []
    if targets:
        # Cada alvo especificado precisa corresponder a pelo menos um módulo elegível
        matched_union: set[str] = set()
        for t in targets:
            matches_for_t = [name for name in eligible_linears if _matches_target(name, t)]
            if not matches_for_t:
                raise LoRAError(f"nenhum módulo nn.Linear elegível corresponde ao alvo {t!r}")
            matched_union.update(matches_for_t)

        selected_names = [name for name in eligible_linears if name in matched_union]
        tied_names = _find_tied_weight_modules(model)
        if any(name in tied_names for name in selected_names):
            raise LoRAError("alvo compartilha pesos; fusão exige um adaptador que preserve esse compartilhamento")
    else:
        # targets vazio: exclui get_output_embeddings e tensores compartilhados (tied)
        output_emb = None
        if hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
            output_emb = model.get_output_embeddings()

        tied_names = _find_tied_weight_modules(model)

        for name, mod in eligible_linears.items():
            if output_emb is not None and mod is output_emb:
                continue
            if name in tied_names:
                continue
            selected_names.append(name)

    if not selected_names:
        raise LoRAError("nenhum módulo nn.Linear elegível encontrado para adaptação LoRA")

    replacements = {name: LoRALinear(eligible_linears[name], rank=rank, alpha=alpha)
                    for name in selected_names}
    # Allocation failures also leave the caller's modules and flags untouched.
    for param in model.parameters():
        param.requires_grad = False

    # 8. Substituição das camadas selecionadas por LoRALinear
    for name in selected_names:
        lora_mod = replacements[name]
        _replace_module(model, name, lora_mod)

    # 9. Contagem de parâmetros treináveis e congelados
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return {
        "modules": selected_names,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def merge_lora(model: Any) -> int:
    """Restaura as camadas nn.Linear originais sequencialmente e funde os pesos LoRA.

    A fusão é calculada em blocos de no máximo 128 linhas para evitar alocação de
    uma matriz FP32 intermediária de tamanho completo, preserva o dtype original
    e confere valores finitos. Chamadas repetidas não têm efeito e retornam 0.
    """
    if not TORCH_AVAILABLE:
        raise LoRAError("PyTorch é necessário para merge_lora")

    lora_modules: list[tuple[str, LoRALinear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            lora_modules.append((name, mod))

    if not lora_modules:
        return 0

    chunk_size = 128

    # Check every chunk before committing any update. A failed merge must not
    # leave some matrices updated and others still wrapped in adapters.
    with torch.no_grad():
        for name, lora_mod in lora_modules:
            weight = lora_mod.base_layer.weight
            for start in range(0, weight.shape[0], chunk_size):
                merged = (weight[start:start + chunk_size].float()
                          + (lora_mod.lora_B[start:start + chunk_size] @ lora_mod.lora_A) * lora_mod.scaling).to(weight.dtype)
                if not torch.isfinite(merged).all():
                    raise LoRAError(f"valores não finitos ao fundir LoRA em {name!r}")

    for name, lora_mod in lora_modules:
        base_layer = lora_mod.base_layer
        A = lora_mod.lora_A.data
        B = lora_mod.lora_B.data
        scaling = lora_mod.scaling

        base_weight = base_layer.weight.data
        out_features, in_features = base_weight.shape

        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            B_chunk = B[start:end]
            delta_chunk_fp32 = (B_chunk @ A) * scaling

            w_chunk = base_weight[start:end]
            merged_chunk = (w_chunk.to(torch.float32) + delta_chunk_fp32).to(base_weight.dtype)

            if not torch.isfinite(merged_chunk).all():
                raise LoRAError(
                    f"valores não finitos encontrados ao fundir LoRA no módulo {name!r} "
                    f"(linhas {start}:{end})"
                )

            base_weight[start:end].copy_(merged_chunk)

        if not torch.isfinite(base_layer.weight).all():
            raise LoRAError(f"pesos resultantes contêm valores não finitos no módulo {name!r}")

        _replace_module(model, name, base_layer)

    return len(lora_modules)

"""Função de perda para destilação via divergência KL com top-k logits e cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def kd_loss(
    student_logits: "torch.Tensor",
    teacher_values: "torch.Tensor",
    teacher_indices: "torch.Tensor",
    labels: "torch.Tensor | None" = None,
    *,
    alpha: float = 0.9,
    temperature: float = 2.0,
    ignore_index: int = -100,
) -> "torch.Tensor":
    """Calcula a perda combinada de destilação (KL sobre top-k) e entropia cruzada supervisionada.

    Args:
        student_logits: [B, T, V] — logits completos do student.
        teacher_values: [B, T, K] — top-k valores de logit do teacher.
        teacher_indices: [B, T, K] — índices correspondentes no vocabulário.
        labels: [B, T] — tokens alvo para o termo supervisionado (opcional).
        alpha: peso do termo de destilação (em [0, 1]).
        temperature: temperatura de escalonamento das distribuições.
        ignore_index: índice ignorado no cálculo da entropia cruzada.
    """
    import torch
    import torch.nn.functional as F

    selected = student_logits.gather(-1, teacher_indices.long())
    s_logp = F.log_softmax(selected / temperature, dim=-1)
    t_prob = F.softmax(teacher_values.float() / temperature, dim=-1)

    loss_kd = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)

    if labels is None or alpha >= 1.0:
        return loss_kd

    shift_logits = student_logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * loss_kd + (1.0 - alpha) * loss_ce

"""Perda de destilação sobre o top-k do teacher.

Guardar os logits completos do teacher é inviável: para 10 mil amostras de 512
tokens com vocabulário de 152 mil, seriam ~1,4 TB. Guardando só os k maiores
valores e seus índices, cai para poucos GB — e a cauda da distribuição
contribui quase nada para a divergência.

Consequência matemática, dita abertamente: a KL passa a ser entre as duas
distribuições **renormalizadas no suporte top-k**, não sobre o vocabulário
inteiro. É a prática padrão em destilação com top-k, e o viés é pequeno porque
o suporte descartado tem massa desprezível.
"""

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
    """KL sobre o top-k do teacher, somada à entropia cruzada supervisionada.

    Args:
        student_logits: [B, T, V] — vocabulário completo.
        teacher_values: [B, T, K] — os k maiores logits do teacher.
        teacher_indices: [B, T, K] — os índices correspondentes em V.
        labels: [B, T] para o termo supervisionado; None desliga-o.
        alpha: peso da destilação. Em **recuperação** fica alto (0,9): o teacher
            é a fonte da verdade e o student já sabe a tarefa.
        temperature: em recuperação fica baixa (2,0) — as distribuições já são
            próximas, e suavizar demais apaga o sinal útil.
    """
    import torch
    import torch.nn.functional as F

    selected = student_logits.gather(-1, teacher_indices.long())
    s_logp = F.log_softmax(selected / temperature, dim=-1)
    t_prob = F.softmax(teacher_values.float() / temperature, dim=-1)

    # O fator T² restaura a magnitude do gradiente, que a divisão por T reduziu
    # em 1/T². Sem ele, `alpha` deixa de significar o que aparenta significar.
    loss_kd = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)

    if labels is None or alpha >= 1.0:
        return loss_kd

    # Shift causal: a posição t prevê o token t+1.
    shift_logits = student_logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * loss_kd + (1.0 - alpha) * loss_ce

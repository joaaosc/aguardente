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
    mask: "torch.Tensor | None" = None,
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
        mask: [B, T] — máscara booleana ou binária indicando tokens válidos (1) e preenchimento (0).
        alpha: peso do termo de destilação (em [0, 1]).
        temperature: temperatura de escalonamento das distribuições.
        ignore_index: índice ignorado no cálculo da entropia cruzada.
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0.0:
        raise ValueError(f"temperature deve ser estritamente positiva, recebido {temperature}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha deve estar no intervalo [0.0, 1.0], recebido {alpha}")

    if labels is not None:
        # O alinhamento causal consome uma posição; com T < 2 não sobra nenhuma
        # transição t -> t + 1 e as reduções abaixo operariam sobre tensores
        # vazios, devolvendo NaN em vez de uma perda.
        if student_logits.size(1) < 2:
            raise ValueError(
                "o alinhamento causal exige pelo menos 2 posições de sequência, "
                f"recebido {student_logits.size(1)}"
            )
        # Alinhamento causal para predição de próximo token:
        # a posição t prediz o token t + 1.
        s_logits = student_logits[:, :-1].contiguous()
        t_vals = teacher_values[:, :-1].contiguous()
        t_idx = teacher_indices[:, :-1].contiguous()
        s_labels = labels[:, 1:].contiguous()
        if mask is not None:
            # Uma transição causal t -> t+1 só é válida se tanto a entrada em t quanto o alvo em t+1 forem tokens reais
            s_mask = (mask[:, :-1].bool() & mask[:, 1:].bool()).contiguous()
        else:
            s_mask = None
    else:
        s_logits = student_logits
        t_vals = teacher_values
        t_idx = teacher_indices
        s_labels = None
        s_mask = mask.bool().contiguous() if mask is not None else None

    # Extrai os logits do aluno correspondentes às top-k escolhas do professor.
    selected = s_logits.gather(-1, t_idx.long())

    # Converte para float32 para estabilidade numérica sob precisão mista ou float16.
    s_logp = F.log_softmax(selected.float() / temperature, dim=-1)
    t_prob = F.softmax(t_vals.float() / temperature, dim=-1)

    # KL divergence per token: soma sobre o top-k renormalizado.
    # clamp(min=0.0) previne desvios negativos ínfimos causados por arredondamento de ponto flutuante.
    kl_per_token = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1).clamp(min=0.0)

    if s_mask is not None:
        pesos = s_mask.to(kl_per_token.dtype)
        loss_kd = (kl_per_token * pesos).sum() / pesos.sum().clamp(min=1.0)
    else:
        loss_kd = kl_per_token.mean()

    # O fator T² restaura a magnitude dos gradientes reduzida pela divisão por T.
    loss_kd = loss_kd * (temperature ** 2)

    if s_labels is None or alpha >= 1.0:
        return loss_kd

    if s_mask is not None:
        s_labels = s_labels.masked_fill(~s_mask, ignore_index)

    if not bool((s_labels != ignore_index).any()):
        loss_ce = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)
    else:
        loss_ce = F.cross_entropy(
            s_logits.reshape(-1, s_logits.size(-1)).float(),
            s_labels.reshape(-1),
            ignore_index=ignore_index,
        )
    return alpha * loss_kd + (1.0 - alpha) * loss_ce

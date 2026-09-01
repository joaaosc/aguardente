"""Destilação de recuperação: devolver ao modelo podado o que o corte tirou."""

from .loss import kd_loss
from .teacher import TeacherLogits, precompute_logits
from .train import RecoveryConfig, RecoveryResult, recover

__all__ = [
    "kd_loss", "TeacherLogits", "precompute_logits",
    "RecoveryConfig", "RecoveryResult", "recover",
]

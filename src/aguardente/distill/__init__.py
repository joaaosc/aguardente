"""Módulo de destilação de recuperação."""

from .loss import kd_loss
from .teacher import TeacherLogits, precompute_logits
from .train import RecoveryConfig, RecoveryResult, recover

__all__ = [
    "kd_loss", "TeacherLogits", "precompute_logits",
    "RecoveryConfig", "RecoveryResult", "recover",
]

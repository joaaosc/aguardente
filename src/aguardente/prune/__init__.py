"""Poda estruturada de LLMs causais no formato transformers."""

from .surgery import PruneReport, prune_model
from .scoring import Scores, score_model

__all__ = ["PruneReport", "prune_model", "Scores", "score_model"]

"""Erros do domínio. Todos carregam uma sugestão acionável quando existe uma."""

from __future__ import annotations


class AguardenteError(Exception):
    """Base. `hint` é o que o usuário pode fazer a respeito."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  → {self.hint}" if self.hint else self.message


class PreflightError(AguardenteError):
    """Ambiente não atende um requisito duro."""


class ProbeError(AguardenteError):
    """Não foi possível sondar o modelo."""


class UnsupportedArchitecture(AguardenteError):
    """A arquitetura não expõe os campos necessários para planejar a poda."""


class PlanImpossible(AguardenteError):
    """O alvo pedido não é alcançável dentro dos limites de poda."""

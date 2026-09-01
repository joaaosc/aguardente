"""Exceções do pacote."""

from __future__ import annotations


class AguardenteError(Exception):
    """Exceção base do pacote, com suporte a sugestão de ação corretiva."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  → {self.hint}" if self.hint else self.message


class PreflightError(AguardenteError):
    """Requisito obrigatório de ambiente não atendido."""


class ProbeError(AguardenteError):
    """Falha ao inspecionar metadados do modelo."""


class UnsupportedArchitecture(AguardenteError):
    """Arquitetura não suportada ou parâmetros insuficientes para o plano de poda."""


class PlanImpossible(AguardenteError):
    """Alvo de parâmetros solicitado não pode ser alcançado dentro dos limites da arquitetura."""

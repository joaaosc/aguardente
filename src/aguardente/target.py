"""Deployment constraints are independent of the preparation machine."""
from dataclasses import asdict, dataclass
import math

from .errors import AguardenteError


@dataclass(frozen=True)
class TargetProfile:
    ram_gib: float = 8.0
    reserve_gib: float = 4.0
    context: int = 2048
    batch_size: int = 1
    max_asset_bytes: int | None = None

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (self.ram_gib, self.reserve_gib))
                or self.ram_gib <= 0 or not 0 <= self.reserve_gib < self.ram_gib):
            raise AguardenteError("RAM e reserva do destino inválidas")
        if self.context < 1 or self.batch_size < 1:
            raise AguardenteError("contexto e batch do destino precisam ser positivos")
        if self.max_asset_bytes is not None and self.max_asset_bytes < 1:
            raise AguardenteError("limite de arquivo precisa ser positivo")

    @property
    def available_bytes(self):
        return int((self.ram_gib - self.reserve_gib) * 1024**3)

    def estimate(self, *, params: int, bits: float, cache_bytes: int = 0,
                 workspace_bytes: int = 0) -> dict:
        if params < 0 or not math.isfinite(bits) or bits <= 0 or min(cache_bytes, workspace_bytes) < 0:
            raise AguardenteError("estimativa de memória inválida")
        weights = math.ceil(params * bits / 8)
        total = weights + cache_bytes + workspace_bytes
        return {"weights_bytes": weights, "cache_bytes": cache_bytes,
                "workspace_bytes": workspace_bytes, "estimated_bytes": total,
                "available_bytes": self.available_bytes,
                "within_estimate": total <= self.available_bytes,
                "runtime_verified": False,
                "note": "estimativa; medir especialização, ativações e memória unificada no destino"}

    def to_dict(self):
        return asdict(self)

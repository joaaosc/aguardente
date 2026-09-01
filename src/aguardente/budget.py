"""Estimativas de uso de memória e capacidade de processamento."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .arch import Arch

GB = 1024 ** 3

# Bits por peso (BPW) estimados para os formatos suportados
BPW_INT4_MACOS = 4.50
BPW_INT4_EMBED_FP16 = 4.56
BPW_FP16 = 16.0

# Margem reservada para o sistema operacional no macOS
SYSTEM_HEADROOM_BYTES = 6 * GB


@dataclass(frozen=True, slots=True)
class Machine:
    ram_bytes: int
    free_disk_bytes: int
    cpu_count: int
    arm64: bool

    @classmethod
    def detect(cls, path: str | Path = "/") -> Machine:
        def _sysctl(key: str, default: int = 0) -> int:
            try:
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=5, check=True).stdout
                return int(out.strip())
            except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                return default

        import platform
        return cls(
            ram_bytes=_sysctl("hw.memsize"),
            free_disk_bytes=shutil.disk_usage(path).free,
            cpu_count=_sysctl("hw.ncpu", 1),
            arm64=platform.machine() == "arm64",
        )

    @property
    def usable_ram_bytes(self) -> int:
        """Memória utilizável após reserva de margem para o sistema."""
        return max(0, self.ram_bytes - SYSTEM_HEADROOM_BYTES)


def weights_bytes(params: int, bits_per_weight: float = BPW_FP16) -> int:
    return int(params * bits_per_weight / 8)


def kv_cache_bytes(a: Arch, seq_len: int, dtype_bytes: int = 2) -> int:
    """Calcula o tamanho do KV cache em bytes para um dado comprimento de sequência."""
    return 2 * a.num_hidden_layers * a.num_key_value_heads * a.head_dim * seq_len * dtype_bytes


def training_bytes(params: int, *, optimizer: str = "adamw", dtype_bytes: int = 2) -> int:
    """Estimativa de memória estática para pesos, gradientes e estados do otimizador."""
    per_param = dtype_bytes * 2  # pesos e gradientes
    per_param += 8 if optimizer == "adamw" else 4
    return int(params * per_param)


@dataclass(frozen=True, slots=True)
class Budget:
    """Orçamento de recursos de máquina para treino e inferência."""

    machine: Machine
    ram_bytes: int
    train_fraction: float = 0.75

    @classmethod
    def for_machine(cls, m: Machine) -> Budget:
        return cls(machine=m, ram_bytes=m.usable_ram_bytes)

    def max_params_for_training(self, *, optimizer: str = "adamw") -> int:
        """Número máximo de parâmetros para treino dentro da memória disponível."""
        per_param = 2 * 2 + (8 if optimizer == "adamw" else 4)
        return int(self.ram_bytes * self.train_fraction / per_param)

    def max_params_for_inference(self, *, bpw: float = BPW_INT4_EMBED_FP16) -> int:
        """Número máximo de parâmetros para inferência dentro da memória disponível."""
        return int(self.ram_bytes * 8 / bpw)

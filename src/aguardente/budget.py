"""Orçamento de máquina: RAM, disco e o que cabe neles.

Tudo derivado por primeiros princípios a partir da arquitetura — nada de
tabelas de desempenho copiadas de blog.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .arch import Arch

GB = 1024 ** 3

# Preset macOS "4bit" do coreai-models: INT4 per-block(32) symmetric_with_clipping.
# 4,50 bits/peso; 4,56 quando as embeddings ficam em fp16 (pesos atados a lm_head).
BPW_INT4_MACOS = 4.50
BPW_INT4_EMBED_FP16 = 4.56
BPW_FP16 = 16.0

# A Apple recomenda deixar esta folga para o sistema no macOS.
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
        """RAM que um processo pode usar sem estrangular o sistema."""
        return max(0, self.ram_bytes - SYSTEM_HEADROOM_BYTES)


def weights_bytes(params: int, bits_per_weight: float = BPW_FP16) -> int:
    return int(params * bits_per_weight / 8)


def kv_cache_bytes(a: Arch, seq_len: int, dtype_bytes: int = 2) -> int:
    """KV cache para um contexto. Escala linearmente com o comprimento —
    é o que costuma estourar a memória em sessão longa, não os pesos."""
    return 2 * a.num_hidden_layers * a.num_key_value_heads * a.head_dim * seq_len * dtype_bytes


def training_bytes(params: int, *, optimizer: str = "adamw", dtype_bytes: int = 2) -> int:
    """Estado fixo de treino, sem ativações.

    AdamW guarda dois momentos em fp32 (8 B/param) além de pesos e gradientes.
    Ativações escalam com batch × seq × hidden × camadas e frequentemente
    dominam — não estão aqui porque dependem de escolhas de execução.
    """
    per_param = dtype_bytes * 2  # pesos + gradientes
    per_param += 8 if optimizer == "adamw" else 4
    return int(params * per_param)


@dataclass(frozen=True, slots=True)
class Budget:
    """Quanto se pode gastar, e o que isso implica em parâmetros."""

    machine: Machine
    ram_bytes: int
    train_fraction: float = 0.75

    @classmethod
    def for_machine(cls, m: Machine) -> Budget:
        return cls(machine=m, ram_bytes=m.usable_ram_bytes)

    def max_params_for_training(self, *, optimizer: str = "adamw") -> int:
        """Maior modelo treinável no orçamento, deixando margem para ativações."""
        per_param = 2 * 2 + (8 if optimizer == "adamw" else 4)
        return int(self.ram_bytes * self.train_fraction / per_param)

    def max_params_for_inference(self, *, bpw: float = BPW_INT4_EMBED_FP16) -> int:
        return int(self.ram_bytes * 8 / bpw)

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

# Margem reservada para o sistema operacional no macOS. Uma constante única não
# serve às duas pontas: 6 GB deixavam um Mac de 8 GB com orçamento irreal e
# desperdiçavam capacidade em máquinas grandes. A fração com piso e teto
# preserva os 6 GB nos 24 GB de referência.
HEADROOM_FRACTION = 0.25
MIN_HEADROOM_BYTES = 4 * GB
MAX_HEADROOM_BYTES = 12 * GB
SYSTEM_HEADROOM_BYTES = 6 * GB  # fallback quando a RAM total é desconhecida


def headroom_bytes(ram_bytes: int) -> int:
    """Margem de sistema proporcional à RAM total, limitada por piso e teto."""
    if ram_bytes <= 0:
        return SYSTEM_HEADROOM_BYTES
    return int(min(max(ram_bytes * HEADROOM_FRACTION, MIN_HEADROOM_BYTES),
                   MAX_HEADROOM_BYTES))


@dataclass(frozen=True, slots=True)
class Machine:
    ram_bytes: int
    free_disk_bytes: int
    cpu_count: int
    arm64: bool

    @classmethod
    def detect(cls, path: str | Path = "/") -> Machine:
        """Mede a máquina; `path` define o volume cujo espaço livre é reportado."""
        def _sysctl(key: str, default: int = 0) -> int:
            try:
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=5, check=True).stdout
                return int(out.strip())
            except (subprocess.SubprocessError, ValueError, FileNotFoundError):
                return default

        # O destino pode ainda não existir: o volume é o do primeiro ancestral
        # existente, e não o de `/`, que descreveria outro disco.
        alvo = Path(path).expanduser().absolute()
        while not alvo.exists() and alvo != alvo.parent:
            alvo = alvo.parent

        import platform
        return cls(
            ram_bytes=_sysctl("hw.memsize"),
            free_disk_bytes=shutil.disk_usage(alvo).free,
            cpu_count=_sysctl("hw.ncpu", 1),
            arm64=platform.machine() == "arm64",
        )

    @property
    def usable_ram_bytes(self) -> int:
        """Memória utilizável após reserva de margem para o sistema."""
        return max(0, self.ram_bytes - headroom_bytes(self.ram_bytes))

    @classmethod
    def other(cls, *, ram_gb: float, disk_gb: float) -> Machine:
        """Uma máquina hipotética, descrita só pelo que entra no cálculo do orçamento.

        Serve para avaliar a viabilidade de um plano em outro computador sem
        executar nada nele — o caso comum é checar se a máquina de um colega
        aguenta o treino antes de levar o modelo até lá. `cpu_count` não entra
        em nenhuma conta de orçamento, e `arm64` é assumido porque o pipeline
        (MPS, Core AI) só roda em Apple Silicon; ficam de fora de propósito,
        em vez de aceitar um valor que não seria usado para nada.
        """
        return cls(ram_bytes=int(ram_gb * GB), free_disk_bytes=int(disk_gb * GB),
                   cpu_count=0, arm64=True)


def weights_bytes(params: int, bits_per_weight: float = BPW_FP16) -> int:
    return int(params * bits_per_weight / 8)


def kv_cache_bytes(a: Arch, seq_len: int, dtype_bytes: int = 2) -> int:
    """Calcula o tamanho do KV cache em bytes para um dado comprimento de sequência."""
    return 2 * a.num_hidden_layers * a.num_key_value_heads * a.head_dim * seq_len * dtype_bytes


def training_bytes(params: int, *, optimizer: str = "adamw", dtype_bytes: int = 2) -> int:
    """Estimativa de memória estática para pesos, gradientes e estados do otimizador.

    Cobre apenas o custo estático. As ativações dependem de lote, comprimento de
    sequência e profundidade, e ficam por conta da margem de `train_fraction`.
    """
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
        if m.ram_bytes <= 0:
            from .errors import InsufficientResources
            raise InsufficientResources(
                "não foi possível medir a RAM desta máquina",
                hint="Verifique o acesso a sysctl hw.memsize. Uma medição indisponível "
                     "não permite calcular um orçamento de treino confiável.",
            )
        return cls(machine=m, ram_bytes=m.usable_ram_bytes)

    def max_params_for_training(self, *, optimizer: str = "adamw",
                                dtype_bytes: int = 2) -> int:
        """Número máximo de parâmetros para treino dentro da memória disponível.

        `dtype_bytes` acompanha o dispositivo: 2 em MPS (float16) e 4 em CPU
        (float32), onde o consumo real dobra em relação ao cálculo padrão.
        """
        per_param = dtype_bytes * 2 + (8 if optimizer == "adamw" else 4)
        return int(self.ram_bytes * self.train_fraction / per_param)

    def max_params_for_inference(self, *, bpw: float = BPW_INT4_EMBED_FP16) -> int:
        """Número máximo de parâmetros para inferência dentro da memória disponível."""
        return int(self.ram_bytes * 8 / bpw)


# Bytes por token de ativação, como múltiplo de hidden_size × dtype_bytes.
#
# Não é uma contagem exata: com SDPA (a implementação de atenção do
# transformers atual), o PyTorch calcula e descarta blocos da matriz de
# atenção em vez de materializar tudo de uma vez, então o custo real por
# token não cresce quadraticamente com seq_len na prática. Mas o autograd
# ainda retém buffers de q/k/v, das duas projeções da MLP e da própria
# atenção para o backward, e parte disso escapa do que o gradient
# checkpointing consegue descartar. O fator é generoso de propósito: errar
# para um lote menor custa tempo; errar para maior derruba a etapa.
_FATOR_ATIVACAO_INFERENCIA = 8
_FATOR_ATIVACAO_TREINO = 16


def suggest_batch_size(*, hidden_size: int, seq_len: int, ram_bytes: int,
                       dtype_bytes: int = 2, reserved_bytes: int = 0,
                       training: bool = False, minimum: int = 1,
                       maximum: int = 32) -> int:
    """Sugere um tamanho de lote que não deveria estourar a RAM disponível.

    `reserved_bytes` é o que já está ocupado antes de processar o primeiro
    lote — normalmente os pesos do modelo carregado. O que sobra é dividido
    pelo custo estimado de um lote de tamanho 1, e o resultado nunca passa de
    `maximum`: um lote muito grande economiza pouco tempo e amplia o risco de
    um travamento tardio, depois de já ter carregado o modelo inteiro.
    """
    margem = ram_bytes - reserved_bytes
    if margem <= 0:
        return minimum
    fator = _FATOR_ATIVACAO_TREINO if training else _FATOR_ATIVACAO_INFERENCIA
    por_amostra = fator * hidden_size * dtype_bytes * seq_len
    if por_amostra <= 0:
        return minimum
    return max(minimum, min(maximum, int(margem // por_amostra)))

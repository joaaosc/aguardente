"""Cálculo analítico da contagem de parâmetros a partir do config do modelo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import UnsupportedArchitecture


@dataclass(frozen=True, slots=True)
class Arch:
    """Dimensões estruturais para contagem de parâmetros e cálculo de poda."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool = True
    # Modelos como Qwen3 aplicam RMSNorm por cabeça em q e k.
    qk_norm: bool = False

    @property
    def heads_per_group(self) -> int:
        """Número de cabeças de query por grupo de chave/valor (GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise UnsupportedArchitecture(
                f"num_attention_heads ({self.num_attention_heads}) não é múltiplo de "
                f"num_key_value_heads ({self.num_key_value_heads})",
                hint="A poda de cabeças requer divisão exata de grupos GQA.",
            )

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any]) -> Arch:
        """Extrai dimensões a partir do config.json."""
        try:
            hidden = int(cfg["hidden_size"])
            heads = int(cfg["num_attention_heads"])
        except KeyError as exc:
            raise UnsupportedArchitecture(
                f"config.json não expõe {exc.args[0]!r}",
                hint="São suportadas apenas arquiteturas de LLM causal no formato transformers.",
            ) from exc

        head_dim = int(cfg.get("head_dim") or hidden // heads)
        n_kv = int(cfg.get("num_key_value_heads") or heads)

        return cls(
            hidden_size=hidden,
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            vocab_size=int(cfg["vocab_size"]),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            qk_norm=str(cfg.get("model_type", "")).startswith("qwen3"),
        )

    def with_(self, **changes: int) -> Arch:
        """Retorna cópia com dimensões alteradas."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class Breakdown:
    """Distribuição de parâmetros por componente."""

    embeddings: int
    lm_head: int
    attention: int
    mlp: int
    norms: int

    @property
    def total(self) -> int:
        return self.embeddings + self.lm_head + self.attention + self.mlp + self.norms

    @property
    def per_layer(self) -> int:
        return self.attention + self.mlp

    def shares(self) -> dict[str, float]:
        t = self.total or 1
        return {
            "embeddings": self.embeddings / t,
            "lm_head": self.lm_head / t,
            "attention": self.attention / t,
            "mlp": self.mlp / t,
            "norms": self.norms / t,
        }


def count_params(a: Arch) -> Breakdown:
    """Calcula a distribuição de parâmetros por componente estrutural.

    Baseado na arquitetura padrão de LLMs causais (atenção com projeções
    q/k/v/o e suporte a GQA, MLP com gate/up/down e camadas de normalização).
    """
    q_out = a.num_attention_heads * a.head_dim
    kv_out = a.num_key_value_heads * a.head_dim

    attn_per_layer = (
        a.hidden_size * q_out      # q_proj
        + a.hidden_size * kv_out   # k_proj
        + a.hidden_size * kv_out   # v_proj
        + q_out * a.hidden_size    # o_proj
    )
    mlp_per_layer = 3 * a.hidden_size * a.intermediate_size  # gate + up + down

    # RMSNorm por cabeça em q e k (Qwen3)
    qk = 2 * a.head_dim * a.num_hidden_layers if a.qk_norm else 0

    embeddings = a.vocab_size * a.hidden_size
    return Breakdown(
        embeddings=embeddings,
        lm_head=0 if a.tie_word_embeddings else embeddings,
        attention=attn_per_layer * a.num_hidden_layers,
        mlp=mlp_per_layer * a.num_hidden_layers,
        norms=2 * a.hidden_size * a.num_hidden_layers + a.hidden_size + qk,
    )


def count_stored(a: Arch, *, lm_head_materialized: bool) -> int:
    """Contagem de parâmetros gravados em disco.

    Leva em conta se a camada lm_head foi materializada como tensor independente.
    """
    total = count_params(a).total
    if a.tie_word_embeddings and lm_head_materialized:
        total += a.vocab_size * a.hidden_size
    return total


def reconcile(a: Arch, reported_total: int | None, *, lm_head_materialized: bool = False) -> float | None:
    """Calcula o erro relativo entre a contagem calculada e o total reportado pelo modelo."""
    if not reported_total:
        return None
    counted = count_stored(a, lm_head_materialized=lm_head_materialized)
    return (counted - reported_total) / reported_total

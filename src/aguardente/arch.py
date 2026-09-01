"""Aritmética de arquitetura: onde os parâmetros de um LLM causal realmente estão.

Puro — sem torch, sem rede. Tudo aqui deriva do `config.json` do Hugging Face,
e é o que permite planejar a poda antes de baixar qualquer peso.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import UnsupportedArchitecture


@dataclass(frozen=True, slots=True)
class Arch:
    """As dimensões que importam para contar parâmetros e podar."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool = True
    # Alguns modelos (Qwen3) aplicam RMSNorm por cabeça em q e k.
    qk_norm: bool = False

    @property
    def heads_per_group(self) -> int:
        """Quantas cabeças de query cada cabeça KV serve (GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise UnsupportedArchitecture(
                f"num_attention_heads ({self.num_attention_heads}) não é múltiplo de "
                f"num_key_value_heads ({self.num_key_value_heads})",
                hint="A poda de cabeças anda em grupos GQA inteiros; sem essa divisão "
                     "exata não há como cortar sem quebrar o agrupamento.",
            )

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any]) -> Arch:
        """Extrai do config.json, tolerando os campos que algumas famílias omitem."""
        try:
            hidden = int(cfg["hidden_size"])
            heads = int(cfg["num_attention_heads"])
        except KeyError as exc:
            raise UnsupportedArchitecture(
                f"config.json não expõe {exc.args[0]!r}",
                hint="Só arquiteturas de LLM causal no formato transformers são suportadas.",
            ) from exc

        # head_dim costuma ser omitido quando é simplesmente hidden / heads.
        head_dim = int(cfg.get("head_dim") or hidden // heads)
        # Sem GQA, num_key_value_heads == num_attention_heads.
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
        """Cópia com dimensões trocadas — usado pelo planejador de poda."""
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
    """Conta parâmetros por componente.

    Modelo de referência: blocos no estilo Llama/Qwen — atenção com projeções
    q/k/v/o (GQA quando num_key_value_heads < num_attention_heads), MLP com
    gate/up/down, e dois RMSNorm por camada mais um final.

    Conta o modelo **lógico**: com embeddings atadas, o lm_head não soma. Para o
    que ocupa disco, use `count_stored()` — não é a mesma grandeza.

    Não inclui biases (a maioria destas famílias não usa).
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

    # RMSNorm por cabeça em q e k (Qwen3): head_dim cada, por camada.
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
    """Parâmetros gravados em disco — o que determina download e tamanho do arquivo.

    Diverge da contagem lógica num ponto que o `config.json` não permite prever:
    alguns modelos gravam `lm_head.weight` mesmo declarando `tie_word_embeddings`
    (o Qwen3-0.6B faz isso; o SmolLM2-135M não). Só o índice de tensores diz.
    """
    total = count_params(a).total
    if a.tie_word_embeddings and lm_head_materialized:
        total += a.vocab_size * a.hidden_size
    return total


def reconcile(a: Arch, reported_total: int | None, *, lm_head_materialized: bool = False) -> float | None:
    """Erro relativo entre a contagem em disco e o total que o Hugging Face reporta.

    Sanidade do planejamento: erro acima de ~1% significa que a arquitetura foge
    do modelo de referência e o plano de poda não é confiável.
    """
    if not reported_total:
        return None
    counted = count_stored(a, lm_head_materialized=lm_head_materialized)
    return (counted - reported_total) / reported_total

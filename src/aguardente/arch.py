"""Cálculo analítico da contagem de parâmetros a partir do config do modelo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .errors import UnsupportedArchitecture

# Chaves que denunciam arquiteturas fora do escopo: a contagem analítica assume
# um decoder causal denso, text-only, com as dimensões no nível de topo.
MOE_KEYS = ("num_experts", "n_routed_experts", "num_local_experts",
            "moe_intermediate_size", "num_experts_per_tok", "n_shared_experts")
VISION_KEYS = ("vision_config", "vision_n_layers", "vision_tower", "visual")

# Atenção latente: q, k e v deixam de ser projeções densas e passam por um posto
# reduzido, que a poda estruturada não sabe fatiar.
MLA_KEYS = ("kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim")

# Camadas de cross-attention intercaladas quebram a uniformidade que a seleção
# de camadas pressupõe.
CROSS_KEYS = ("cross_attention_layers", "cross_attn_layers")

# Chaves genéricas o bastante para colidir com outro uso: só contam como sinal
# de arquitetura multimodal quando carregam uma subconfiguração.
AMBIGUOUS_KEYS = ("visual",)

# Pesos já quantizados chegam com todas as dimensões densas no topo e passariam
# pelas demais guardas, mas a poda estruturada não sabe fatiar seus tensores.
QUANT_DTYPES = ("int8", "uint8", "int4", "uint4", "fp8", "float8")


def _present(cfg: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Chaves com valor significativo. Um zero ou vazio explícito é ausência."""
    achadas = []
    for k in keys:
        v = cfg.get(k)
        if not v:
            continue
        if k in AMBIGUOUS_KEYS and not isinstance(v, dict):
            continue
        achadas.append(k)
    return achadas


def _quantized(cfg: dict[str, Any]) -> str | None:
    """Descreve o esquema de quantização declarado no config, se houver."""
    if cfg.get("quantization_config"):
        q = cfg["quantization_config"]
        metodo = q.get("quant_method") if isinstance(q, dict) else None
        return f"quantization_config ({metodo})" if metodo else "quantization_config"
    dtype = str(cfg.get("dtype") or cfg.get("torch_dtype") or "").lower()
    if any(marca in dtype for marca in QUANT_DTYPES):
        return f"dtype {dtype}"
    return None


def _reject_unsupported(cfg: dict[str, Any], *, origem: str = "") -> None:
    """Recusa decoders cuja contagem densa produziria um resultado silenciosamente errado.

    A guarda opera sobre o config **do decoder**, já desaninhado: um sinal de MoE
    no `language_config` de um modelo multimodal conta tanto quanto um no topo.
    Torre de visão e projetor não são mais motivo de recusa — são descartados
    pela extração —, e por isso `VISION_KEYS` deixou de levantar aqui.
    """
    onde = f"o {origem}" if origem else "o config.json"
    if moe := _present(cfg, MOE_KEYS):
        raise UnsupportedArchitecture(
            f"decoder de texto MoE não suportado ({onde} expõe {', '.join(moe)})",
            hint="A contagem de parâmetros assume uma MLP densa por camada; em um modelo "
                 "MoE ela erraria por ordens de grandeza. Use um decoder denso.",
        )
    if mla := _present(cfg, MLA_KEYS):
        raise UnsupportedArchitecture(
            f"atenção latente (MLA) não suportada ({onde} expõe {', '.join(mla)})",
            hint="Em MLA as projeções de q, k e v passam por um posto reduzido, e a "
                 "poda estruturada fatia projeções densas.",
        )
    if cross := _present(cfg, CROSS_KEYS):
        raise UnsupportedArchitecture(
            f"decoder com camadas de cross-attention ({onde} expõe {', '.join(cross)})",
            hint="A seleção de camadas pressupõe blocos uniformes; intercalar tipos "
                 "diferentes quebraria o padrão que o modelo espera.",
        )
    if quant := _quantized(cfg):
        raise UnsupportedArchitecture(
            f"pesos pré-quantizados não suportados ({onde} expõe {quant})",
            hint="A poda estruturada e a destilação operam sobre tensores densos em "
                 "float16. Use o repositório com os pesos originais não quantizados.",
        )


def _norms_per_layer(cfg: dict[str, Any]) -> int:
    """Quantidade de RMSNorm por camada, que o config.json não declara.

    Gemma 2 e Gemma 3 normalizam também a saída de cada bloco, somando duas
    normas por camada. Nas demais famílias são duas.
    """
    return 4 if str(cfg.get("model_type", "")).startswith("gemma") else 2


# Chaves sob as quais um checkpoint multimodal aninha o config do decoder.
TEXT_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder_config")

# Campos que costumam ficar só no nível de topo e pertencem ao decoder.
INHERITED = ("tie_word_embeddings", "torch_dtype", "dtype", "vocab_size",
             "bos_token_id", "eos_token_id", "pad_token_id")

def text_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Localiza o config do decoder e devolve (config mesclado, chave de origem).

    Quando as dimensões estão no topo, o próprio config é o do decoder — é o
    caso do Qwen2.5-VL. Caso contrário elas estão aninhadas, sob uma chave que
    varia por família.
    """
    if "hidden_size" in cfg:
        return dict(cfg), ""

    achadas = [k for k in TEXT_CONFIG_KEYS if isinstance(cfg.get(k), dict)]
    if not achadas:
        raise UnsupportedArchitecture(
            "o config.json não expõe 'hidden_size' no topo nem aninha as dimensões em "
            + ", ".join(TEXT_CONFIG_KEYS),
            hint="São suportadas arquiteturas de LLM causal no formato transformers.",
        )
    if len(achadas) > 1:
        raise UnsupportedArchitecture(
            "o config.json aninha mais de um sub-config de texto: "
            + ", ".join(achadas),
            hint="Não há como escolher entre eles sem adivinhar.",
        )

    chave = achadas[0]
    # O sub-config tem precedência: quando declara um campo, é ele que vale.
    mesclado = {k: cfg[k] for k in INHERITED if k in cfg} | dict(cfg[chave])
    return mesclado, chave



def _qk_norm(cfg: dict[str, Any]) -> bool:
    """Normalização por cabeça em q e k: chave explícita quando houver, senão heurística."""
    for chave in ("use_qk_norm", "qk_norm", "attention_qk_norm", "qk_layernorm"):
        if chave in cfg:
            return bool(cfg[chave])
    return str(cfg.get("model_type", "")).startswith("qwen3")


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
    # Qwen2 e derivados somam bias em q, k e v. São poucos parâmetros, mas
    # ignorá-los faz a contagem divergir do total publicado sem motivo real.
    attention_bias: bool = False
    # Normas do tamanho de hidden_size por camada. A maioria usa duas, antes da
    # atenção e antes da MLP; a família Gemma usa quatro, com uma norma extra
    # depois de cada bloco.
    norms_per_layer: int = 2

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
        """Extrai dimensões a partir do config.json, desaninhando quando preciso.

        Num modelo multimodal as dimensões do decoder ficam sob `text_config` ou
        equivalente. Descer até elas é o que permite planejar a poda de um VLM
        sem baixar peso algum.
        """
        cfg, origem = text_config(cfg)
        _reject_unsupported(cfg, origem=origem)
        try:
            hidden = int(cfg["hidden_size"])
            heads = int(cfg["num_attention_heads"])
            intermediate = int(cfg["intermediate_size"])
            layers = int(cfg["num_hidden_layers"])
            vocab = int(cfg["vocab_size"])
        except KeyError as exc:
            raise UnsupportedArchitecture(
                f"config.json não expõe {exc.args[0]!r}",
                hint="São suportadas apenas arquiteturas de LLM causal no formato transformers.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise UnsupportedArchitecture(
                f"config.json tem dimensão com valor inválido: {exc}",
                hint="As dimensões do config.json precisam ser inteiros.",
            ) from exc

        if not cfg.get("head_dim") and hidden % heads:
            raise UnsupportedArchitecture(
                f"hidden_size ({hidden}) não é múltiplo de num_attention_heads ({heads}) "
                "e o config.json não declara head_dim",
                hint="Sem head_dim explícito a dimensão por cabeça seria truncada, e a "
                     "contagem de parâmetros sairia errada sem qualquer sinal.",
            )
        head_dim = int(cfg.get("head_dim") or hidden // heads)
        n_kv = int(cfg.get("num_key_value_heads") or heads)

        return cls(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            vocab_size=vocab,
            # O padrão segue o do transformers (True); divergir daqui erraria a
            # contagem pelo tamanho inteiro do tensor de embeddings.
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
            qk_norm=_qk_norm(cfg),
            attention_bias=bool(cfg.get("attention_bias", False)),
            norms_per_layer=_norms_per_layer(cfg),
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

    # Bias de q, k e v quando a arquitetura os usa. o_proj não tem bias.
    bias_per_layer = q_out + 2 * kv_out if a.attention_bias else 0

    # RMSNorm por cabeça em q e k (Qwen3)
    qk = 2 * a.head_dim * a.num_hidden_layers if a.qk_norm else 0

    embeddings = a.vocab_size * a.hidden_size
    return Breakdown(
        embeddings=embeddings,
        lm_head=0 if a.tie_word_embeddings else embeddings,
        attention=(attn_per_layer + bias_per_layer) * a.num_hidden_layers,
        mlp=mlp_per_layer * a.num_hidden_layers,
        norms=a.norms_per_layer * a.hidden_size * a.num_hidden_layers + a.hidden_size + qk,
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

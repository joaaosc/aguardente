"""Dimensionamento do alvo e plano de poda.

A ordem dos eixos não é arbitrária: vem da distribuição real de parâmetros.
Num LLM causal típico o MLP é ~2/3 do total, a atenção ~1/4, e as embeddings
o resto. Cortar onde há massa é o que dá retorno.

Ordem, e a razão de cada posição:

1. `intermediate_size` — maior fatia, e o corte é local: cada FFN é
   independente. Alinhado a múltiplos de 128 para que a compressão per-block
   não pule camadas por indivisibilidade.
2. cabeças de atenção — em grupos GQA inteiros, nunca por cabeça solta.
3. camadas — o corte mais brutal por parâmetro removido: descarta uma
   transformação inteira do residual stream.

`hidden_size` fica fora: podá-lo obriga a fatiar embeddings, lm_head, todas as
projeções e normalizações de forma coerente, e é a maior fonte de bugs sutis.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arch import Arch, count_params
from .errors import PlanImpossible

# Pisos por eixo, como fração do original. Além disto a degradação deixa de ser
# recuperável por destilação em tempo razoável.
FLOOR_INTERMEDIATE = 0.25
FLOOR_KV_GROUPS = 0.50
FLOOR_LAYERS = 0.50

# Fatias do escalar de aperto t ∈ [0,1] atribuídas a cada eixo, na ordem.
_STAGE_INTERMEDIATE = 0.50
_STAGE_HEADS = 0.30
_STAGE_LAYERS = 0.20

ALIGN = 128


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """De → para, por eixo."""

    source: Arch
    target: Arch
    requested_params: int

    @property
    def source_params(self) -> int:
        return count_params(self.source).total

    @property
    def target_params(self) -> int:
        return count_params(self.target).total

    @property
    def ratio(self) -> float:
        return self.source_params / max(1, self.target_params)

    @property
    def is_noop(self) -> bool:
        return self.source == self.target

    def changes(self) -> list[tuple[str, int, int]]:
        """Eixos que mudaram, como (nome, de, para)."""
        fields = ("intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "num_hidden_layers")
        return [
            (f, getattr(self.source, f), getattr(self.target, f))
            for f in fields
            if getattr(self.source, f) != getattr(self.target, f)
        ]


def _lerp_int(hi: int, lo: int, t: float, *, align: int = 1, minimum: int = 1) -> int:
    """Interpola de `hi` até `lo` conforme t ∈ [0,1], alinhado e com piso."""
    t = min(1.0, max(0.0, t))
    raw = hi - (hi - lo) * t
    v = int(round(raw / align)) * align if align > 1 else int(round(raw))
    return max(minimum, min(hi, v))


def shrink(a: Arch, t: float, *, align: int = ALIGN) -> Arch:
    """Aplica um aperto t ∈ [0,1] respeitando a ordem de prioridade dos eixos.

    Para t pequeno só o `intermediate_size` encolhe; as cabeças só começam a
    cair depois que ele atinge o piso, e as camadas por último.
    """
    t = min(1.0, max(0.0, t))

    t_int = min(1.0, t / _STAGE_INTERMEDIATE)
    t_hd = min(1.0, max(0.0, t - _STAGE_INTERMEDIATE) / _STAGE_HEADS)
    t_ly = min(1.0, max(0.0, t - _STAGE_INTERMEDIATE - _STAGE_HEADS) / _STAGE_LAYERS)

    inter = _lerp_int(
        a.intermediate_size, int(a.intermediate_size * FLOOR_INTERMEDIATE),
        t_int, align=align, minimum=align,
    )
    groups = _lerp_int(
        a.num_key_value_heads, max(1, int(a.num_key_value_heads * FLOOR_KV_GROUPS)),
        t_hd, minimum=1,
    )
    layers = _lerp_int(
        a.num_hidden_layers, max(1, int(a.num_hidden_layers * FLOOR_LAYERS)),
        t_ly, minimum=1,
    )

    return a.with_(
        intermediate_size=inter,
        num_key_value_heads=groups,
        num_attention_heads=groups * a.heads_per_group,  # GQA preservado
        num_hidden_layers=layers,
    )


def plan_for_target(a: Arch, target_params: int, *, align: int = ALIGN) -> PrunePlan:
    """Menor aperto que atinge o alvo, por busca binária sobre t.

    `shrink` é monotónico decrescente em t, então a busca é bem definida.
    """
    if target_params <= 0:
        raise PlanImpossible("alvo de parâmetros precisa ser positivo")

    if count_params(a).total <= target_params:
        return PrunePlan(source=a, target=a, requested_params=target_params)

    floor = shrink(a, 1.0, align=align)
    if count_params(floor).total > target_params:
        raise PlanImpossible(
            f"alvo de {target_params/1e9:.2f} B é menor que o piso de poda "
            f"({count_params(floor).total/1e9:.2f} B)",
            hint="As embeddings sozinhas podem já exceder o alvo. Escolha um modelo "
                 "menor, ou aceite um alvo maior.",
        )

    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if count_params(shrink(a, mid, align=align)).total > target_params:
            lo = mid
        else:
            hi = mid

    return PrunePlan(source=a, target=shrink(a, hi, align=align),
                     requested_params=target_params)

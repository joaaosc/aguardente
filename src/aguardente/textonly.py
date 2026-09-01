"""Localização do decoder causal dentro de um checkpoint.

Modelos multimodais publicam o decoder de texto junto da torre de visão e do
projetor, sob um prefixo que varia por família: `language_model.model.` no
DeepSeek-VL e no LLaVA, `model.language_model.` no Qwen3-VL, `model.text_model.`
no SmolVLM, e nada no Qwen2.5-VL, que já grava o decoder na raiz.

Uma tabela por arquitetura envelheceria a cada modelo novo. O prefixo é
descoberto pela estrutura: procura-se o grupo de tensores `…layers.N.…` cujos
sufixos cobrem a assinatura de um bloco transformer denso. Só o decoder a
cobre — o projetor não tem atenção, e as torres de visão nomeiam suas camadas
de outra forma.

Este módulo não abre arquivo nem depende de torch: opera sobre os cabeçalhos
já lidos, o que o torna testável sem rede e sem pesos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import prod
from typing import Any, Iterator, Mapping

from .errors import UnsupportedArchitecture

# Raiz canônica de um decoder causal no formato transformers.
ROOT = "model."

_LAYER = re.compile(r"^(?P<prefixo>.*?)layers\.(?P<indice>\d+)\.(?P<sufixo>.+)$")

# Sufixos que um bloco transformer denso sempre tem. A poda estruturada fatia
# cada uma dessas projeções separadamente, então exigir o conjunto completo
# recusa de saída o que ela não saberia cortar: projeções fundidas (`qkv_proj`
# do Phi-3, `query_key_value` do Falcon) e nomenclaturas de fora do padrão
# (`attention.wqkv` do InternLM2).
SIGNATURE = frozenset({
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
})

# Tensores do decoder que ficam fora da lista de camadas.
EMBED = "embed_tokens.weight"
NORM = "norm.weight"
LM_HEAD = "lm_head.weight"


@dataclass(frozen=True, slots=True)
class TextLayout:
    """Mapa do decoder causal dentro de um checkpoint possivelmente multimodal."""

    prefix: str
    num_layers: int
    rename: dict[str, str] = field(repr=False)
    dropped: tuple[str, ...] = field(repr=False)
    kept_params: int = 0
    dropped_params: int = 0
    lm_head: str | None = None

    @property
    def kept(self) -> tuple[str, ...]:
        return tuple(self.rename)

    @property
    def is_multimodal(self) -> bool:
        """Há tensores fora do decoder — torre de visão, projetor, adaptadores."""
        return bool(self.dropped)

    @property
    def needs_extraction(self) -> bool:
        """A extração muda alguma coisa: descarta tensores ou renomeia chaves."""
        return self.is_multimodal or any(o != n for o, n in self.rename.items())

    @property
    def tie_word_embeddings(self) -> bool:
        """Sem `lm_head` materializado, o checkpoint compartilha as embeddings."""
        return self.lm_head is None


def _tensors(headers: Mapping[str, Mapping[str, Any]]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Percorre os tensores de todos os shards, recusando nome repetido.

    O mesmo nome em dois shards não é ambiguidade a resolver por precedência: é
    sinal de checkpoint inconsistente, e escolher um dos dois em silêncio
    produziria pesos errados sem qualquer aviso.
    """
    visto: dict[str, str] = {}
    for arquivo, header in headers.items():
        for nome, meta in header.items():
            if nome == "__metadata__":
                continue
            if nome in visto:
                raise UnsupportedArchitecture(
                    f"tensor {nome!r} aparece em {visto[nome]} e em {arquivo}",
                    hint="O checkpoint está inconsistente. Baixe o repositório novamente.",
                )
            visto[nome] = arquivo
            yield nome, meta


def _numel(meta: Mapping[str, Any]) -> int:
    return prod(meta.get("shape") or ()) if meta.get("shape") else 0


def _candidatos(camadas: dict[str, dict[int, set[str]]]) -> list[str]:
    """Prefixos cujos sufixos, somados sobre as camadas, cobrem a assinatura."""
    return [p for p, idx in camadas.items()
            if SIGNATURE <= set().union(*idx.values())]


def _sem_candidato(camadas: dict[str, dict[int, set[str]]]) -> UnsupportedArchitecture:
    """Erro que diz qual prefixo chegou mais perto e o que faltou nele."""
    if not camadas:
        return UnsupportedArchitecture(
            "nenhum tensor do checkpoint segue o padrão `…layers.N.…`",
            hint="São suportados decoders causais densos no formato transformers.",
        )
    faltas = {p: SIGNATURE - set().union(*idx.values()) for p, idx in camadas.items()}
    melhor = min(faltas, key=lambda p: len(faltas[p]))
    ausentes = ", ".join(sorted(faltas[melhor]))
    fundidas = {"self_attn.q_proj.weight", "mlp.gate_proj.weight"} & faltas[melhor]
    hint = ("A poda estruturada fatia q/k/v e gate/up separadamente. Modelos que "
            "fundem essas projeções — Phi-3, Falcon, InternLM2 — ainda não são "
            "suportados.") if fundidas else (
            "São suportados decoders causais densos no formato transformers.")
    return UnsupportedArchitecture(
        f"não foi encontrado um decoder causal denso; o prefixo {melhor!r} chegou "
        f"mais perto e não tem: {ausentes}",
        hint=hint,
    )


def _conferir_camadas(prefixo: str, indices: dict[int, set[str]]) -> int:
    """Exige camadas contíguas e homogêneas, e devolve quantas são."""
    total = max(indices) + 1
    if set(indices) != set(range(total)):
        faltando = sorted(set(range(total)) - set(indices))
        raise UnsupportedArchitecture(
            f"a lista de camadas em {prefixo!r} tem buracos: falta "
            f"{', '.join(str(i) for i in faltando[:5])}",
            hint="Índices não contíguos indicam camadas de tipos intercalados.",
        )
    # Camadas com conjuntos de sufixos diferentes denunciam intercalamento de
    # tipos — cross-attention a cada N blocos, como no mllama. Selecionar
    # camadas nesse caso quebraria o padrão que o modelo espera.
    referencia = indices[0]
    for i in sorted(indices):
        if indices[i] != referencia:
            divergentes = ", ".join(sorted(indices[i] ^ referencia))
            raise UnsupportedArchitecture(
                f"a camada {i} de {prefixo!r} difere da camada 0 em: {divergentes}",
                hint="O decoder precisa ser uniforme para que a seleção de camadas "
                     "preserve o comportamento do modelo.",
            )
    return total


def _conferir_vocabulario(prefixo: str, kept: Mapping[str, str], total: int) -> None:
    """Recusa tensor sob o prefixo que não seja parte reconhecida do decoder.

    Descartar em silêncio um tensor que pertence ao decoder é o defeito que só
    apareceria como perplexidade ruim, horas depois, na etapa de recuperação.
    """
    aceitos = {f"{ROOT}{EMBED}", f"{ROOT}{NORM}", LM_HEAD}
    for original, canonico in kept.items():
        if canonico in aceitos:
            continue
        if _LAYER.match(canonico) and canonico.startswith(f"{ROOT}layers."):
            indice = int(_LAYER.match(canonico)["indice"])
            if indice < total:
                continue
        raise UnsupportedArchitecture(
            f"tensor inesperado sob o decoder {prefixo!r}: {original}",
            hint="Só embeddings, camadas, norma final e lm_head são reconhecidos.",
        )


def discover_layout(headers: Mapping[str, Mapping[str, Any]]) -> TextLayout:
    """Localiza o decoder causal e monta o mapa de renomeação para a raiz canônica."""
    camadas: dict[str, dict[int, set[str]]] = {}
    nomes: dict[str, int] = {}
    cabecas: list[str] = []

    for nome, meta in _tensors(headers):
        nomes[nome] = _numel(meta)
        if nome.endswith(LM_HEAD):
            cabecas.append(nome)
        if m := _LAYER.match(nome):
            idx = camadas.setdefault(m["prefixo"], {})
            idx.setdefault(int(m["indice"]), set()).add(m["sufixo"])

    achados = _candidatos(camadas)
    if not achados:
        raise _sem_candidato(camadas)
    if len(achados) > 1:
        raise UnsupportedArchitecture(
            "mais de um decoder causal no checkpoint: "
            + ", ".join(repr(p) for p in sorted(achados)),
            hint="Não há como escolher entre eles sem adivinhar. Use o repositório "
                 "de um modelo com um único decoder.",
        )

    prefixo = achados[0]
    total = _conferir_camadas(prefixo, camadas[prefixo])

    for obrigatorio in (EMBED, NORM):
        if f"{prefixo}{obrigatorio}" not in nomes:
            raise UnsupportedArchitecture(
                f"o decoder em {prefixo!r} não expõe {obrigatorio}",
                hint="Embeddings e norma final são necessárias para reconstruir o modelo.",
            )
    if len(cabecas) > 1:
        raise UnsupportedArchitecture(
            "mais de um lm_head no checkpoint: " + ", ".join(sorted(cabecas)),
            hint="Não há como escolher a cabeça de saída sem adivinhar.",
        )

    # O lm_head pode ficar fora do prefixo: o SmolVLM o grava na raiz, enquanto o
    # decoder mora em `model.text_model.`. Por isso ele é localizado por sufixo, e
    # não por prefixo.
    cabeca = cabecas[0] if cabecas else None

    rename: dict[str, str] = {}
    dropped: list[str] = []
    mantidos = descartados = 0
    for nome, numel in nomes.items():
        if nome == cabeca:
            rename[nome] = LM_HEAD
            mantidos += numel
        elif nome.startswith(prefixo):
            rename[nome] = ROOT + nome[len(prefixo):]
            mantidos += numel
        else:
            dropped.append(nome)
            descartados += numel

    _conferir_vocabulario(prefixo, rename, total)

    return TextLayout(
        prefix=prefixo,
        num_layers=total,
        rename=rename,
        dropped=tuple(sorted(dropped)),
        kept_params=mantidos,
        dropped_params=descartados,
        lm_head=cabeca,
    )

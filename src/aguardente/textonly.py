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

from .arch import Arch, count_stored
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


# Chaves sob as quais um checkpoint multimodal aninha o config do decoder.
TEXT_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder_config")

# Campos que costumam ficar só no nível de topo e pertencem ao decoder.
INHERITED = ("tie_word_embeddings", "torch_dtype", "dtype", "vocab_size",
             "bos_token_id", "eos_token_id", "pad_token_id")

# Número de cabeças assumido pela classe de config do transformers quando o
# config aninhado o omite. É hipótese, nunca conclusão: o valor derivado daqui
# é obrigatoriamente conferido contra as formas de q_proj e k_proj, e uma
# divergência interrompe. O DeepSeek-VL depende disto — seu `language_config`
# declara só camadas e vocabulário.
CLASS_HEADS = {"llama": 32, "mistral": 32}


def text_config(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
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
            "o config.json não expõe as dimensões no topo nem as aninha em "
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


def _forma(formas: Mapping[str, list[int]], nome: str) -> list[int]:
    forma = formas.get(nome)
    if not forma:
        raise UnsupportedArchitecture(
            f"o checkpoint não traz a forma de {nome}",
            hint="O cabeçalho do safetensors precisa declarar `shape` para cada tensor.",
        )
    return forma


def _cabecas(cfg: Mapping[str, Any], *, q_out: int, kv_out: int,
             hidden: int) -> tuple[int, int, int]:
    """Resolve (head_dim, cabeças de query, grupos de chave/valor).

    As formas dão `q_out = cabeças × head_dim`, que tem mais de uma fatoração.
    A ordem de precedência resolve a ambiguidade sem adivinhar: head_dim
    declarado, senão número de cabeças declarado, senão o default da classe —
    e o resultado é sempre conferido contra as formas.
    """
    if cfg.get("head_dim"):
        head_dim = int(cfg["head_dim"])
        origem = "head_dim do config"
    elif cfg.get("num_attention_heads"):
        cabecas = int(cfg["num_attention_heads"])
        if q_out % cabecas:
            raise UnsupportedArchitecture(
                f"q_proj tem {q_out} linhas, que não é múltiplo das "
                f"{cabecas} cabeças declaradas no config",
                hint="O config não corresponde aos pesos do checkpoint.",
            )
        head_dim = q_out // cabecas
        origem = "num_attention_heads do config"
    else:
        tipo = str(cfg.get("model_type") or "")
        if tipo not in CLASS_HEADS:
            raise UnsupportedArchitecture(
                f"o config não declara head_dim nem num_attention_heads, e não há "
                f"default conhecido para model_type {tipo!r}",
                hint="Informe as dimensões no config do decoder de texto.",
            )
        head_dim = q_out // CLASS_HEADS[tipo]
        origem = f"default da classe {tipo}"

    if head_dim <= 0 or q_out % head_dim or kv_out % head_dim:
        raise UnsupportedArchitecture(
            f"head_dim {head_dim} ({origem}) não divide as projeções observadas: "
            f"q_proj tem {q_out} linhas e k_proj tem {kv_out}",
            hint="As formas dos pesos são a verdade; o config diverge delas.",
        )
    cabecas, grupos = q_out // head_dim, kv_out // head_dim
    if cabecas % grupos:
        raise UnsupportedArchitecture(
            f"{cabecas} cabeças não são múltiplo de {grupos} grupos de chave/valor",
            hint="A poda de cabeças requer divisão exata de grupos GQA.",
        )
    return head_dim, cabecas, grupos


def arch_from_shapes(headers: Mapping[str, Mapping[str, Any]], layout: TextLayout, *,
                     cfg: Mapping[str, Any] | None = None) -> Arch:
    """Reconstrói as dimensões a partir das formas dos tensores.

    Quando o config está incompleto — como no DeepSeek-VL, cujo `language_config`
    omite hidden_size, intermediate_size e o número de cabeças — as formas são a
    única fonte confiável. O config entra apenas onde as formas são ambíguas, e
    mesmo aí o resultado é conferido contra elas.
    """
    formas: dict[str, list[int]] = {}
    for arquivo, header in headers.items():
        for nome, meta in header.items():
            if canonico := layout.rename.get(nome):
                formas[canonico] = list(meta.get("shape") or ())

    vocab, hidden = _forma(formas, f"{ROOT}{EMBED}")[:2]
    intermediate = _forma(formas, f"{ROOT}layers.0.mlp.gate_proj.weight")[0]
    q_out = _forma(formas, f"{ROOT}layers.0.self_attn.q_proj.weight")[0]
    kv_out = _forma(formas, f"{ROOT}layers.0.self_attn.k_proj.weight")[0]

    head_dim, cabecas, grupos = _cabecas(cfg or {}, q_out=q_out, kv_out=kv_out,
                                         hidden=hidden)

    # A quantidade de normas por camada não está em config algum: o Gemma 3
    # normaliza também a saída de cada bloco e tem quatro, contra as duas do
    # Llama. Contá-las pelo sufixo `layernorm.weight` acerta as duas famílias
    # sem precisar saber de qual se trata, e exclui as normas de q e k, que
    # terminam em `q_norm.weight` e são contadas à parte. A conferência de forma
    # evita confundi-las com o bias de q_proj, que no Qwen2 e no InternVL3 tem
    # exatamente o tamanho de hidden_size.
    normas = sum(1 for nome, forma in formas.items()
                 if nome.startswith(f"{ROOT}layers.0.")
                 and nome.endswith("layernorm.weight") and forma == [hidden])

    return Arch(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layout.num_layers,
        num_attention_heads=cabecas,
        num_key_value_heads=grupos,
        head_dim=head_dim,
        vocab_size=vocab,
        tie_word_embeddings=layout.tie_word_embeddings,
        norms_per_layer=normas,
        qk_norm=f"{ROOT}layers.0.self_attn.q_norm.weight" in formas,
        attention_bias=f"{ROOT}layers.0.self_attn.q_proj.bias" in formas,
    )


def reconcile(arch: Arch, layout: TextLayout) -> int:
    """Diferença entre a contagem analítica e a soma real dos tensores mantidos.

    Zero é o único resultado aceitável antes de escrever qualquer byte: uma
    divergência significa que a `Arch` não descreve os pesos que serão gravados,
    e o plano de poda calculado sobre ela cortaria as dimensões erradas.
    """
    return count_stored(arch, lm_head_materialized=not layout.tie_word_embeddings) \
        - layout.kept_params

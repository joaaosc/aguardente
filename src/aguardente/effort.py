"""Níveis de esforço da destilação.

Um nível de esforço decide **quanto trabalho** se investe para recuperar a
qualidade perdida na poda: quantas amostras de calibração, quantos logits do
teacher são pré-computados, com que profundidade de top-k, e por quantas épocas
o aluno treina. Não decide *quanto cortar* — isso continua sendo `--target-params`,
que é um eixo ortogonal: o alvo define o tamanho do resultado, o esforço define
o cuidado com que se chega nele.

A precedência é sempre do usuário: uma opção informada explicitamente na linha
de comando vence o preset, e o painel diz quais foram sobrescritas.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator

from .errors import AguardenteError

# Opções que o nível de esforço define. `batch_size` fica de fora de propósito:
# é restrição de memória da máquina, não escolha de qualidade.
CONTROLLED = ("calib_batches", "seq_len", "logit_batches", "top_k",
              "epochs", "lr", "alpha", "temperature", "grad_accum")

DEFAULT = "medium"


@dataclass(frozen=True, slots=True)
class Effort:
    """Preset de agressividade da destilação."""

    name: str
    label: str
    summary: str
    when: str

    # calibração e poda
    calib_batches: int
    seq_len: int

    # logits do teacher
    logit_batches: int
    top_k: int

    # treino de recuperação
    epochs: int
    lr: float
    alpha: float
    temperature: float
    grad_accum: int

    def values(self) -> dict[str, Any]:
        return {campo: getattr(self, campo) for campo in CONTROLLED}

    def logit_bytes(self, batch_size: int) -> int:
        """Espaço em disco dos logits pré-computados, em bytes."""
        from .distill.teacher import estimate_logit_bytes

        amostras = self.logit_batches * batch_size
        return estimate_logit_bytes(amostras, self.seq_len, top_k=self.top_k)[0]

    @property
    def work(self) -> float:
        """Índice de trabalho: tokens processados, ponderados por etapa.

        A pontuação da poda e a pré-computação de logits são forward puro; o
        treino soma forward e backward, com peso 3. O número só tem significado
        relativo, e é isso que se mostra ao usuário.
        """
        pontuacao = self.calib_batches * self.seq_len
        logits = self.logit_batches * self.seq_len
        treino = self.epochs * self.logit_batches * self.seq_len * 3
        return float(pontuacao + logits + treino)


LEVELS: tuple[Effort, ...] = (
    Effort(
        name="low", label="rápido",
        summary="Passada curta: valida o caminho inteiro sem investir tempo em qualidade.",
        when="Para testar o pipeline, o ambiente ou um modelo novo antes de gastar horas.",
        calib_batches=8, seq_len=256,
        logit_batches=64, top_k=64,
        epochs=1, lr=5e-5, alpha=0.8, temperature=2.0, grad_accum=2,
    ),
    Effort(
        name="medium", label="equilibrado",
        summary="Configuração de referência: recupera boa parte da queda a um custo previsível.",
        when="O padrão. Serve à maioria das conversões de modelos de 1 B a 4 B.",
        calib_batches=32, seq_len=512,
        logit_batches=256, top_k=128,
        epochs=2, lr=3e-5, alpha=0.9, temperature=2.0, grad_accum=4,
    ),
    Effort(
        name="high", label="cuidadoso",
        summary="Mais amostras, top-k mais fundo e uma época extra, com passo menor.",
        when="Quando o resultado vai ser usado de verdade e há tempo de máquina disponível.",
        calib_batches=64, seq_len=768,
        logit_batches=512, top_k=192,
        epochs=3, lr=2e-5, alpha=0.9, temperature=2.0, grad_accum=8,
    ),
    Effort(
        name="max", label="exaustivo",
        summary="Tudo no limite: sequência longa, top-k largo e quatro épocas de passo curto.",
        when="Poda agressiva que perdeu muita qualidade, ou entrega final sem pressa.",
        calib_batches=128, seq_len=1024,
        logit_batches=1024, top_k=256,
        epochs=4, lr=1.5e-5, alpha=0.95, temperature=2.0, grad_accum=8,
    ),
)

NAMES = tuple(e.name for e in LEVELS)
_BY_NAME = {e.name: e for e in LEVELS}


def get(name: str | None) -> Effort:
    """Resolve o nível pelo nome, com erro acionável quando desconhecido."""
    chave = (name or DEFAULT).strip().lower()
    if chave not in _BY_NAME:
        raise AguardenteError(
            f"nível de esforço desconhecido: {name!r}",
            hint=f"Use um de: {', '.join(NAMES)}. Execute `aguardente effort` para ver "
                 "o que cada nível muda.",
        )
    return _BY_NAME[chave]


def index(effort: Effort) -> int:
    """Posição do nível na escala, de 0 (rápido) a 3 (exaustivo)."""
    return NAMES.index(effort.name)


def relative_cost(effort: Effort, *, base: str = DEFAULT) -> float:
    """Custo de tempo relativo a outro nível, pelo índice de trabalho."""
    return effort.work / get(base).work


def resolve(effort: Effort, informado: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Combina o preset com as opções informadas na linha de comando.

    Devolve os valores finais e a lista de nomes que o usuário sobrescreveu.
    O preset nunca vence uma escolha explícita: silenciar a opção de quem a
    digitou seria o mesmo defeito que reaproveitar estado divergente.
    """
    valores = effort.values()
    sobrescritos = []
    for campo, valor in informado.items():
        if campo in valores and valor is not None:
            valores[campo] = valor
            sobrescritos.append(campo)
    return valores, sorted(sobrescritos)


def comparison(batch_size: int = 2) -> Iterator[dict[str, Any]]:
    """Linhas da tabela comparativa entre os níveis."""
    for e in LEVELS:
        yield {
            "name": e.name,
            "label": e.label,
            "epochs": e.epochs,
            "top_k": e.top_k,
            "seq_len": e.seq_len,
            "logit_batches": e.logit_batches,
            "logit_bytes": e.logit_bytes(batch_size),
            "cost": relative_cost(e),
        }


def describe(effort: Effort) -> list[tuple[str, str]]:
    """Campos do preset em pares (rótulo, valor) para exibição."""
    return [
        ("lotes de calibração", f"{effort.calib_batches}"),
        ("comprimento de sequência", f"{effort.seq_len} tokens"),
        ("lotes de logits", f"{effort.logit_batches}"),
        ("profundidade top-k", f"{effort.top_k}"),
        ("épocas", f"{effort.epochs}"),
        ("taxa de aprendizado", f"{effort.lr:g}"),
        ("peso da destilação (alpha)", f"{effort.alpha}"),
        ("acumulação de gradiente", f"{effort.grad_accum}"),
    ]


assert {f.name for f in fields(Effort)} >= set(CONTROLLED), "preset incompleto"

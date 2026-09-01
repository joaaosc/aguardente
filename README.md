# aguardente

Encolhe modelos de linguagem para caberem no seu Mac — e converte o resultado
para o formato nativo da Apple.

Pega um modelo do Hugging Face grande demais para a sua máquina, **remove
parâmetros** por poda estruturada, **recupera a qualidade** perdida por
destilação, e entrega um `.aimodel` pronto para rodar via Core AI.

<!--
  MÍDIA — substituir pelos arquivos reais em docs/media/.
  Sugestão de captura:
    asciinema rec docs/media/demo.cast --cols 80 --rows 30
    agg docs/media/demo.cast docs/media/demo.gif --font-size 16
-->
![Demonstração do pipeline completo](docs/media/demo.gif)
*(placeholder — gravar `aguardente run` do início ao fim)*

---

## Instalação

Um comando. Resolve Homebrew, `uv`, `aria2`, o interpretador Python correto e
o próprio programa:

```bash
curl -fsSL https://raw.githubusercontent.com/SEU-USUARIO/aguardente/main/install.sh | bash
```

Ou, a partir de um clone do repositório:

```bash
git clone https://github.com/SEU-USUARIO/aguardente && cd aguardente && ./install.sh
```

O instalador é idempotente: rodar de novo apenas atualiza o que mudou.

> **Por que um instalador e não `pip install`?** O `python3` que vem no PATH de
> um Mac costuma ser 3.13 ou 3.14, e o stack Core AI exige `>=3.11,<3.14`. Um
> `pip install` direto falha na resolução de dependências com uma mensagem
> difícil de interpretar. O script fixa o interpretador e evita o problema.

---

## Uso

```bash
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.4e9 --measure
```

Um comando faz tudo e **retoma de onde parou** se for interrompido. Para ver o
plano antes de baixar qualquer coisa:

```bash
aguardente plan Qwen/Qwen3-4B
```

<!-- placeholder: captura estática da saída de `aguardente plan` -->
![Saída do comando plan](docs/media/plan.png)

Guia completo, com todos os parâmetros e o que fazer quando algo falha:
**[usage.md](usage.md)**.

---

## Como funciona

Duas reduções diferentes acontecem, e confundi-las é o erro conceitual mais
comum:

|  | O que reduz | Custo | Quem faz |
|---|---|---|---|
| **Poda + destilação** | número de **parâmetros** | horas de treino | este projeto |
| **Compressão** | **bits por peso** | minutos, sem dados | ferramenta da Apple |

Quantizar um modelo de 4 B para int4 dá um arquivo 4× menor com **4 B de
parâmetros ainda lá**. Só a redução estrutural muda a contagem. O pipeline usa
as duas, nesta ordem.

### As cinco etapas

```
fetch  →  prune  →  logits  →  recover  →  export
 rede    minutos   dezenas    horas      minutos
                   de min
```

| Etapa | O que faz |
|---|---|
| **fetch** | Baixa o modelo original com `aria2` — retomável, várias conexões |
| **prune** | Mede a importância de cada unidade e corta MLP, cabeças e camadas |
| **logits** | Pré-computa as previsões do original e o descarrega da memória |
| **recover** | Treina o modelo podado a imitar o original |
| **export** | Converte para `.aimodel` via a ferramenta oficial da Apple |

### O que se corta, e por quê nessa ordem

A ordem vem da distribuição real de parâmetros. Num Qwen3-4B: embeddings
9,7 %, atenção 23,5 %, **MLP 66,8 %**.

1. **`intermediate_size`** — a maior fatia, e o corte é local. Alinhado a
   múltiplos de 128 para que a compressão posterior não pule camadas.
2. **Cabeças de atenção** — em grupos GQA inteiros, nunca por cabeça solta.
3. **Camadas** — o corte mais brutal por parâmetro removido. Primeira e última
   nunca saem: camadas de fronteira são desproporcionalmente sensíveis.

`hidden_size` fica de fora: podá-lo obriga a fatiar embeddings, `lm_head`,
todas as projeções e normalizações de forma coerente.

A importância é medida num conjunto de calibração pequeno — norma L2 das
ativações para MLP e grupos KV, *block influence* (`1 − cos(entrada, saída)`)
para camadas.

---

## Resultado típico

Qwen3-4B com alvo de 1,4 B parâmetros:

|  | original | podado | recuperado + int4 |
|---|---|---|---|
| parâmetros | 4,02 B | 1,37 B | 1,37 B |
| em disco | 7,49 GB | 2,55 GB | **0,73 GB** |
| redução | — | 2,9× | **10,2×** |

Com `--measure`, a perplexidade é medida em três pontos — original, pós-poda e
recuperado — e o relatório informa que fração da queda foi recuperada.

---

## Requisitos

| Item | Exigência |
|---|---|
| Sistema | macOS 27+ com Apple Silicon |
| Xcode | 27+ com Metal Toolchain |
| RAM | 24 GB para modelos de 4 B; 16 GB para alvos menores |
| Disco | ~30 GB livres |

`aguardente doctor` verifica tudo e diz o que fazer quando algo falta.

---

## O que este projeto não faz

A Apple já resolve a conversão e a compressão através do `coreai.llm.export`,
e este projeto **invoca** essa ferramenta em vez de reimplementá-la. O que ela
não oferece, e que aqui existe:

- **Poda e destilação** — não existem em nenhum pacote do stack Core AI.
- **Medição de perplexidade** — o `coreai.llm.eval` da Apple é um esqueleto
  que apenas informa que a funcionalidade virá depois.
- **Orquestração retomável** com estado em disco.

---

## Estado

| Componente | Estado |
|---|---|
| `doctor`, `plan`, `fetch`, `status` | funcionando, testados |
| `run` (pipeline completo) | funcionando ponta a ponta, retomável |
| `export` | argumentos validados; execução real ainda não exercitada |

93 testes automatizados:

```bash
uv run --with pytest --with torch --with transformers pytest
```

---

## Licença

Ver `LICENSE`. Os utilitários em `src/aguardente/vendor/` vêm do projeto
`apple/coreai-models` sob BSD-3-Clause, com a atribuição preservada.

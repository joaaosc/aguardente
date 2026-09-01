# Guia de uso

Documento de referência para comandos, opções e resolução de problemas.

- [Diagnóstico do ambiente](#diagnóstico-do-ambiente)
- [Execução de teste](#execução-de-teste)
- [Comandos](#os-comandos)
- [Detalhamento do pipeline](#detalhamento-do-pipeline)
- [Retomada de execução](#retomada-de-execução)
- [Otimização de memória](#otimização-de-memória)
- [Resolução de problemas](#resolução-de-problemas)
- [Perguntas frequentes](#perguntas-frequentes)

---

## Diagnóstico do ambiente

```bash
aguardente doctor
```

Executa a verificação dos pré-requisitos necessários para as ferramentas do pipeline:

| Verificação | Motivo |
|---|---|
| macOS 27+ | Suporte ao framework Core AI |
| Xcode 27+ | Compilador de modelos (`coreai-build`) |
| `coreai-build` | Ferramenta do Metal Toolchain |
| `aria2c` | Downloads acelerados e retomáveis |
| Python 3.11–3.13 | Compatibilidade com o stack Core AI |
| Apple Silicon | Arquitetura de execução suportada (`arm64`) |
| Dependências do pipeline | `torch`, `transformers` e pacotes Core AI |
| Recursos (RAM e disco) | Estimativa de espaço e memória disponíveis |

---

## Execução de teste

Para validar o fluxo completo e as dependências em poucos minutos usando um modelo leve:

```bash
aguardente run HuggingFaceTB/SmolLM2-135M-Instruct \
    -o ensaio --target-params 90e6 \
    --calib-batches 4 --logit-batches 8 --epochs 1 \
    --export-dry-run
```

A opção `--export-dry-run` valida os argumentos e a configuração de conversão sem executar o empacotamento completo.

---

## Os comandos

### `plan` — Planejamento analítico

```bash
aguardente plan Qwen/Qwen3-4B [--target-params 1.4e9]
```

Analisa a arquitetura do modelo a partir do `config.json` e dos metadados remotos sem baixar os pesos. Exibe a distribuição de parâmetros, as dimensões sugeridas após a poda e as estimativas de memória e disco.

Caso `--target-params` não seja especificado, o alvo é sugerido com base na RAM da máquina.

### `fetch` — Download isolado

```bash
aguardente fetch Qwen/Qwen3-4B -o teacher/ [--connections 8]
```

Baixa apenas os arquivos necessários para o modelo (configurações, tokenizers e pesos em `.safetensors`), ignorando variantes e formatos não utilizados.

### `run` — Execução completa

```bash
aguardente run <modelo> -o <diretório> [opções]
```

| Grupo | Opções |
|---|---|
| Alvo | `--target-params 1.4e9` |
| Download | `--connections 8` `--concurrent 4` |
| Calibração | `--calib-batches 32` `--batch-size 2` `--seq-len 512` `--calib-dataset` `--calib-file` |
| Logits | `--logit-batches 256` `--top-k 128` |
| Recuperação | `--epochs 2` `--lr 3e-5` `--alpha 0.9` `--temperature 2.0` `--grad-accum 4` |
| Exportação | `--platform macOS` `--compression 4bit` `--max-context-length` `--export-dry-run` |
| Controle | `--device` `--measure` `--skip-recover` `--skip-export` `--skip-checks` |

A opção `--measure` calcula a perplexidade no início, pós-poda e após a recuperação para quantificar a fração recuperada.

### `status` — Inspeção de progresso

```bash
aguardente status -o run/qwen3
```

Exibe o estado das etapas registradas em `state.json`.

---

## Detalhamento do pipeline

### 1. fetch

Baixa os arquivos do modelo original via `aria2c`.

### 2. prune

Carrega o modelo, avalia a importância estrutural com base em amostras reais de calibração e realiza o corte in-place dos pesos (MLP, grupos de atenção GQA e camadas). Valida que a saída após a poda permanece finita.

### 3. logits

Executa o modelo original sobre o conjunto de calibração para pré-computar os top-k logits por posição e gravá-los em shards no disco. Em seguida, libera a memória ocupada pelo modelo teacher.

### 4. recover

Treina o modelo podado para minimizar a divergência KL em relação aos logits do teacher combinada com a entropia cruzada. O treinamento inclui verificação periódica de perda/perplexidade e parada antecipada caso haja estagnação.

### 5. export

Gera o pacote `.aimodel` chamando `coreai.llm.export` com a configuração de plataforma e compressão selecionadas.

---

## Retomada de execução

O progresso é mantido no arquivo `state.json` no diretório de saída. Caso a execução seja interrompida, reiniciar o mesmo comando pula as etapas já finalizadas:

```bash
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.4e9
```

Estrutura de arquivos gerada:

```
run/qwen3/
├── state.json      # estado e métricas das etapas
├── teacher/        # arquivos do modelo original
├── pruned/         # pesos e configs após poda estruturada
├── logits/         # shards de logits pré-computados
├── student/        # modelo recuperado após destilação
├── ckpt/           # checkpoints de treinamento
└── bundle/         # artefato .aimodel final
```

---

## Otimização de memória

Se o treinamento exceder a capacidade de memória da máquina, considere as seguintes opções:

| Opção | Impacto |
|---|---|
| `--grad-accum 8` | Mantém o tamanho efetivo de batch reduzindo a memória por passo |
| `--seq-len 256` | Reduz o volume de ativações durante o backward pass |
| `--batch-size 1` | Menor consumo de memória por lote |
| `--target-params <menor>` | Gera um modelo menor, exigindo menos memória no treino |
| `--skip-recover` | Pula a etapa de destilação de recuperação |

---

## Resolução de problemas

### Ambiente

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `Core AI exige macOS 27.0 ou superior` | Versão de sistema incompatível | Atualize o macOS para exportação de `.aimodel`. |
| `coreai-build não encontrado` | Metal Toolchain ausente | Execute `xcodebuild -downloadComponent MetalToolchain`. |
| `aria2c não encontrado` | Utilitário não instalado | Instale via `brew install aria2`. |
| `coreai-opt exige >=3.11,<3.14` | Versão do Python incompatível | Crie o ambiente virtual com Python 3.12 (`uv venv --python 3.12`). |
| `coreai-core só publica wheels para macOS arm64` | Arquitetura Intel | O runtime exige Macs com Apple Silicon. |

### Download

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `aria2c saiu com código N` | Falha de conexão | Reexecute o comando; o download continuará do ponto atual. |
| `N arquivo(s) incompletos após o download` | Arquivo corrompido ou truncado | Reexecute o comando para completar os arquivos ausentes. |
| `acesso negado` / `Modelo gated` | Licença não aceita no Hub | Aceite os termos na página do modelo e execute `hf auth login`. |
| `não publica pesos em .safetensors` | Formato não suportado | O pipeline suporta apenas modelos distribuídos em `.safetensors`. |

### Poda e treinamento

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `o modelo podado produz NaN ou Inf` | Poda excessiva desestabilizou o modelo | Aumente o valor de `--target-params`. |
| `alvo é menor que o piso de poda` | Alvo abaixo do limite estrutural | Escolha um modelo base menor ou eleve `--target-params`. |
| `num_attention_heads não é múltiplo de num_key_value_heads` | Arquitetura não padrão | Incompatível com o corte em grupos GQA. |
| `MPS backend out of memory` | Memória de GPU esgotada | Use `--grad-accum 8` e `--seq-len 256`. |

### Exportação

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `coreai.llm.export não encontrado` | Dependência não instalada | Execute o instalador `./install.sh` ou instale o pacote correspondente. |
| `coreai.llm.export falhou (código N)` | Parâmetros de exportação rejeitados | Execute com `--export-dry-run` para inspecionar os argumentos. |

---

## Perguntas frequentes

**Qual é a diferença entre poda e quantização?**
A quantização reduz a precisão dos pesos (por exemplo, de 16 bits para 4 bits), reduzindo o tamanho em disco e a memória de inferência sem alterar a contagem de parâmetros. A poda remove parâmetros estruturalmente (linhas e colunas de matrizes e camadas inteiras). O pipeline combina poda estruturada com quantização final para maximizar a redução.

**Por que a perplexidade sobe imediatamente após a poda?**
A remoção de parâmetros afeta temporariamente a capacidade de representação do modelo. A etapa de destilação subsequente ajusta os pesos restantes para restaurar a qualidade.

**A execução pode ser interrompida?**
Sim. Ao interromper com `Ctrl-C`, o estado atual é salvo em `state.json` e checkpoints de treinamento são persistidos para retomada posterior.

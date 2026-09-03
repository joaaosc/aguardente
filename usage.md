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

Quando uma verificação falha, a instrução exibida ao lado já indica a correção. Para inspecionar a saída bruta do comando que falhou:

```bash
aguardente doctor --verbose
```

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

## Identificando o modelo

Todo comando que recebe um modelo — `plan`, `fetch`, `run`, `extract` — aceita três formas equivalentes:

```bash
aguardente plan Qwen/Qwen3-4B                              # identificador namespace/nome
aguardente plan https://huggingface.co/Qwen/Qwen3-4B       # URL colada do navegador
aguardente plan ./meu-modelo-baixado                        # diretório local já existente
```

**URL do Hugging Face.** É pura normalização: `namespace/nome` já está na URL, com ou sem `/tree/main`, `/blob/...` ou barra final no fim. Não há consulta de rede extra para isso.

**URL do GitHub.** Muitos autores de modelo publicam o código no GitHub sob o mesmo nome que usam no Hugging Face, e é comum copiar a URL errada por hábito. `owner/repo` da URL do GitHub é tratado como uma *aposta* de identificador do Hugging Face, e a aposta só é aceita depois de confirmada por uma chamada à API — nunca por adivinhação:

```bash
aguardente plan https://github.com/Qwen/Qwen3-4B
#   origem       https://github.com/Qwen/Qwen3-4B (GitHub) → Qwen/Qwen3-4B (Hugging Face)
#                outros modelos do mesmo autor no Hugging Face: Qwen/Qwen3-8B, Qwen/Qwen3-14B, ...
```

A linha de origem aparece sempre que a URL foi traduzida, para que fique claro o que o comando vai de fato baixar. Quando o `owner/repo` do GitHub **não** existe no Hugging Face, o comando para — nunca baixa o modelo errado — e mostra dois grupos, cada um rotulado pelo que realmente é:

```
erro: 'algum-autor/Projeto' não existe no Hugging Face.
  repositórios do autor 'algum-autor':
    algum-autor/Projeto-7B
    algum-autor/Projeto-13B-Instruct
  → Rode novamente com um dos identificadores acima, no formato namespace/nome.
```

Nenhuma dessas listas vem de busca aproximada por IA ou de heurística de nome: são exatamente os resultados que a API de metadados do Hugging Face devolve para o autor e para o termo de busca, sem reordenar por "parecença" nem preencher lacunas. Se a API não devolver nada, o erro diz isso — não inventa um candidato.

### `plan` — Planejamento analítico

```bash
aguardente plan Qwen/Qwen3-4B [--target-params 1.4e9]
```

Analisa a arquitetura do modelo a partir do `config.json` e dos metadados remotos sem baixar os pesos. Exibe a distribuição de parâmetros, as dimensões sugeridas após a poda e as estimativas de memória e disco.

Caso `--target-params` não seja especificado, o alvo é sugerido com base na RAM da máquina.

**Viabilidade em outra máquina.** `--other-ram-gb` e `--other-disk-gb` (sempre juntos) avaliam se a conversão e a destilação caberiam num computador diferente do atual, sem tocar nele — útil para checar, antes de levar o modelo até lá, se a máquina de um colega aguenta o treino:

```bash
aguardente plan Qwen/Qwen3-4B --other-ram-gb 24 --other-disk-gb 480
```

A seção "Outra máquina" mostra dois vereditos separados, porque são perguntas diferentes:

- **conversão (poda + export)** depende de disco: baixar o checkpoint original, gravar a cópia podada e o bundle comprimido, tudo cabendo ao mesmo tempo;
- **destilação (recuperação)** depende de RAM de treino: o *piso de poda* da arquitetura — o menor tamanho que os limites por eixo permitem alcançar — precisa caber no teto de treino da máquina. É essa comparação, não o tamanho do modelo em si, que decide se a etapa `recover` é executável ali.

Quando a destilação não é viável, a mensagem aponta `--skip-recover` como saída: a máquina ainda pode podar e exportar o modelo, só não treina a recuperação de qualidade.

### `fetch` — Download isolado

```bash
aguardente fetch Qwen/Qwen3-4B -o teacher/ [--connections 8]
```

Baixa apenas os arquivos necessários para o modelo (configurações, tokenizers e pesos em `.safetensors`), ignorando variantes e formatos não utilizados.

### `extract` — Decoder de texto de um modelo multimodal

Extrai o decoder causal de um checkpoint de visão e linguagem e grava um repositório `transformers` comum, utilizável fora do aguardente.

```bash
aguardente extract deepseek-ai/deepseek-vl-7b-chat -o run/dsvl
aguardente extract ./teacher -o run/dsvl              # de um diretório já baixado
```

Escreve `run/dsvl/teacher-text/` com os pesos do decoder, um `config.json` de arquitetura causal, e o tokenizer. A torre de visão, o projetor e o processador de imagem ficam de fora.

A cópia é feita no nível dos bytes, sem carregar o modelo: funciona sem `torch` e preserva qualquer precisão, `bfloat16` inclusive.

`--discard-source-weights` consome os shards de origem à medida que os processa, o que reduz o pico de disco pela metade em troca de exigir novo download para refazer a etapa.

A mesma extração acontece dentro do `run`, como etapa entre o download e a poda; o comando avulso existe para inspecionar o resultado antes de investir horas de treino.

### `run` — Execução completa

```bash
aguardente run <modelo> -o <diretório> [opções]
```

| Grupo | Opções |
|---|---|
| Esforço | `--effort low\|medium\|high\|max` |
| Alvo | `--target-params 1.0e9` `--allow-oversized` |
| Download | `--connections 8` `--concurrent 4` |
| Calibração | `--calib-batches 32` `--batch-size 2` `--seq-len 512` `--calib-dataset` `--calib-file` |
| Logits | `--logit-batches 256` `--top-k 128` |
| Recuperação | `--epochs 2` `--lr 3e-5` `--alpha 0.9` `--temperature 2.0` `--grad-accum 4` |
| Exportação | `--platform macOS` `--compression 4bit` `--max-context-length` `--export-dry-run` |
| Controle | `--device` `--measure` `--skip-recover` `--skip-export` `--skip-checks` `--restart` |

A opção `--measure` calcula a perplexidade no início, pós-poda e após a recuperação para quantificar a fração recuperada.

O alvo é validado contra o teto de treino da máquina antes de qualquer download. `--allow-oversized` aceita um alvo acima do teto, assumindo o risco de esgotar a memória na recuperação.

O diretório de execução pertence a um par (modelo, alvo). Apontar `-o` para um diretório criado com outro modelo ou outro alvo interrompe a execução em vez de reaproveitar artefatos incompatíveis; `--restart` descarta o estado e os subdiretórios `teacher/`, `pruned/`, `logits/`, `student/`, `bundle/` e `ckpt/` para recomeçar do zero.

Cada etapa grava a impressão digital dos parâmetros que a produziram. Retomar com `--top-k`, `--seq-len`, `--batch-size` ou os hiperparâmetros de treino diferentes refaz as etapas afetadas, em vez de reaproveitar um resultado gerado sob outra configuração.

Duas execuções simultâneas sobre o mesmo `-o` são recusadas por um lock no diretório. Um lock deixado por processo interrompido é recuperado automaticamente.

### `effort` — Escala de agressividade da destilação

```bash
aguardente effort
```

Compara os quatro níveis sem tocar em nenhum modelo: o que cada um muda, quando usar, quanto ocupa em disco e quanto custa em tempo relativo.

| Nível | Perfil | Épocas | top-k | Sequência | Logits em disco | Custo |
|---|---|---:|---:|---:|---:|---:|
| `low` | rápido | 1 | 64 | 256 | 12 MB | 0,07× |
| `medium` | equilibrado | 2 | 128 | 512 | 192 MB | 1,00× |
| `high` | cuidadoso | 3 | 192 | 768 | 864 MB | 4,26× |
| `max` | exaustivo | 4 | 256 | 1.024 | 3,0 GB | 14,74× |

O nível define `--calib-batches`, `--seq-len`, `--logit-batches`, `--top-k`, `--epochs`, `--lr`, `--alpha`, `--temperature` e `--grad-accum`. Qualquer uma dessas opções informada explicitamente tem precedência sobre o preset, e o painel de esforço lista as que foram sobrescritas.

`--batch-size` não entra na escala: é restrição de memória da máquina, não escolha de qualidade.

O nível participa da impressão digital das etapas — trocar de `--effort` entre execuções refaz a poda, os logits e a recuperação, em vez de reaproveitar resultados gerados sob outra configuração.

### `status` — Inspeção de progresso

```bash
aguardente status -o run/qwen3
```

Exibe o estado das etapas registradas em `state.json`.

---

## Detalhamento do pipeline

### 1. fetch

Baixa os arquivos do modelo original via `aria2c`.

### 2. extract

Localiza o decoder causal dentro do checkpoint e grava um modelo só de texto. Pulada quando o modelo já é um decoder comum. Confere a contagem de parâmetros do que gravou contra a arquitetura calculada, e valida a configuração de exportação com `--dry-run`.

### 3. prune

Carrega o modelo, avalia a importância estrutural com base em amostras reais de calibração e realiza o corte in-place dos pesos (MLP, grupos de atenção GQA e camadas). Valida que a saída após a poda permanece finita.

**De onde vem o student.** Por padrão, o student não é uma arquitetura nova: é o próprio teacher, com pedaços cortados fora — mesmos pesos herdados, mesmo tokenizer, só `intermediate_size`, cabeças de atenção e camadas menores. A cirurgia (`prune/surgery.py`) fatia as matrizes existentes; a escolha do que cortar vem de `prune/scoring.py`, que mede num forward real sobre dados de calibração a magnitude de ativação de cada neurônio da MLP e de cada grupo de atenção, e a similaridade de cosseno entre entrada e saída de cada camada — uma camada quase-identidade contribui pouco e é a primeira candidata a sair, com a primeira e a última sempre protegidas.

**Um student pronto.** `--student <modelo>` substitui essa cirurgia por um modelo já existente — um identificador do Hugging Face ou um diretório local. Nesse caso não há alvo de parâmetros a calcular: a etapa `prune` baixa (ou copia) o modelo indicado no lugar de podar, e a recuperação treina esse student contra os logits do teacher, exatamente como faria com o resultado da poda.

A única exigência é o `vocab_size` do student bater com o do teacher, checada antes de qualquer download: a destilação por logits compara probabilidade token a token, e um índice que aponta para palavras diferentes em cada tokenizer invalida a divergência KL. Vocabulário igual não garante o mesmo tokenizer — é uma condição necessária, verificável sem baixar peso algum, não uma prova completa de compatibilidade.

```bash
aguardente run Qwen/Qwen3-4B -o run/custom --student Qwen/Qwen2.5-1.5B-Instruct --skip-recover
```

### 4. logits

Executa o modelo original sobre o conjunto de calibração para pré-computar os top-k logits por posição e gravá-los em shards no disco. Em seguida, libera a memória ocupada pelo modelo teacher.

### 5. recover

Treina o modelo podado para minimizar a divergência KL em relação aos logits do teacher combinada com a entropia cruzada. O treinamento inclui verificação periódica de perda/perplexidade e parada antecipada caso haja estagnação.

### 6. export

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
| `Xcode: Command Line Tools ativo` | `xcode-select` aponta para o CLT, não para o Xcode | Execute `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`. |
| `Xcode: licença não aceita` | Termos do Xcode pendentes | Execute `sudo xcodebuild -license accept`. |
| `Xcode: instalação incompleta` | Componentes adicionais não instalados | Execute `xcodebuild -runFirstLaunch`. |
| `Xcode: sem resposta` | Primeira execução do `xcodebuild` demorada | Execute `xcodebuild -version` no Terminal e repita o diagnóstico. |
| `Xcode: diretório ativo inexistente` | O caminho selecionado sumiu após atualizar ou mover o Xcode | Execute `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`. |
| `sudo: comando bloqueado` em máquina gerenciada | Política de MDM impede alterar o diretório ativo | Exporte `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` no shell. |
| `stack do pipeline` como falha no `run` | `torch` ou pacotes Core AI ausentes | Instale com `uv pip install 'aguardente[pipeline]'`; no `doctor` a mesma condição é apenas um aviso. |
| `coreai-build não encontrado` | Metal Toolchain ausente | Execute `xcodebuild -downloadComponent MetalToolchain`. |
| `aria2c não encontrado` | Utilitário não instalado | Instale via `brew install aria2`. |
| `coreai-opt exige >=3.11,<3.14` | Versão do Python incompatível | Crie o ambiente virtual com Python 3.12 (`uv venv --python 3.12`). |
| `coreai-core só publica wheels para macOS arm64` | Arquitetura Intel | O runtime exige Macs com Apple Silicon. |

### Download

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `aria2c saiu com código N` | Falha de conexão | Reexecute o comando; o download continuará do ponto atual. |
| `espaço insuficiente para ...` | Volume de destino não comporta a etapa | Libere espaço ou aponte `-o` para outro volume; a verificação ocorre antes da escrita. |
| `disco cheio durante a escrita` | Volume esgotado em tempo de execução | Libere espaço e retome; o estado da execução é preservado. |
| `pertence a outra execução` | `-o` aponta para o diretório de outro modelo ou alvo | Use outro diretório, ou `--restart` para descartar o anterior. |
| `outra execução já está usando` | Duas instâncias sobre o mesmo `-o` | Aguarde a conclusão, ou remova `run.lock` se o processo não existe mais. |
| `state.json está corrompido` | Escrita interrompida ou arquivo editado | Remova o arquivo para reiniciar, ou use outro diretório. |
| `N arquivo(s) incompletos após o download` | Arquivo corrompido ou truncado | Reexecute o comando para completar os arquivos ausentes. |
| `acesso negado` / `Modelo gated` | Licença não aceita no Hub | Aceite os termos na página do modelo e execute `hf auth login`. |
| `não publica pesos em .safetensors` | Formato não suportado | O pipeline suporta apenas modelos distribuídos em `.safetensors`. |
| `arquitetura MoE não suportada` | Modelo com mistura de especialistas | A contagem de parâmetros assume MLP densa; escolha um decoder denso. |
| `arquitetura multimodal não suportada` | Modelo com torre de visão | O pipeline poda e destila apenas o decoder causal de texto. |
| `config.json aninha as dimensões em ...` | Config de modelo multimodal | Use o repositório do decoder de texto correspondente. |

### Poda e treinamento

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `o modelo podado produz NaN ou Inf` | Poda excessiva desestabilizou o modelo | Aumente o valor de `--target-params`. |
| `alvo é menor que o piso de poda` | Alvo abaixo do limite estrutural | Escolha um modelo base menor ou eleve `--target-params`. |
| `num_attention_heads não é múltiplo de num_key_value_heads` | Arquitetura não padrão | Incompatível com o corte em grupos GQA. |
| `MPS backend out of memory` | Memória de GPU esgotada | Use `--grad-accum 8` e `--seq-len 256`. |
| `excede o teto de treino desta máquina` | Alvo maior do que a RAM comporta no treino | Reduza `--target-params`, use `--skip-recover`, ou `--allow-oversized` para assumir o risco. |
| `memória insuficiente para concluir a etapa` | Falta de memória durante a execução | Reduza o alvo, o lote ou o comprimento de sequência, e aumente `--grad-accum`. |
| `pesos pré-quantizados não suportados` | Repositório distribui pesos GPTQ, AWQ ou FP8 | Use o repositório com os pesos originais em float16. |
| `não é múltiplo de num_attention_heads e o config.json não declara head_dim` | Dimensão por cabeça indeterminada | Use um modelo cujo `config.json` declare `head_dim`. |

### Exportação

| Sintoma | Causa provável | Ação recomendada |
|---|---|---|
| `coreai.llm.export não encontrado` | Dependência não instalada | Execute o instalador `./install.sh` ou instale o pacote correspondente. |
| `coreai.llm.export falhou (código N)` | Parâmetros de exportação rejeitados | Execute com `--export-dry-run` para inspecionar os argumentos. |

---

### Animações

Etapas longas exibem indicadores que se atualizam no lugar: um medidor de nível ao apresentar o esforço, um spinner durante o carregamento do modelo e a medição de importância, e barras de progresso com estimativa de término no download e na pré-computação dos logits.

A animação só acontece em terminal interativo. Sob `--json`, em pipe ou em arquivo de log, a saída degrada para marcos textuais a cada 10%, sem sequências de escape. Para desligar num terminal, defina `AGUARDENTE_NO_ANIM=1`.

### Depuração

Falhas inesperadas são reportadas com mensagem e ação sugerida, sem traceback. Para investigar a origem, defina `AGUARDENTE_DEBUG=1` e repita o comando:

```bash
AGUARDENTE_DEBUG=1 aguardente run <modelo> -o run/
```

---

## Perguntas frequentes

**Qual é a diferença entre poda e quantização?**
A quantização reduz a precisão dos pesos (por exemplo, de 16 bits para 4 bits), reduzindo o tamanho em disco e a memória de inferência sem alterar a contagem de parâmetros. A poda remove parâmetros estruturalmente (linhas e colunas de matrizes e camadas inteiras). O pipeline combina poda estruturada com quantização final para maximizar a redução.

**Por que a perplexidade sobe imediatamente após a poda?**
A remoção de parâmetros afeta temporariamente a capacidade de representação do modelo. A etapa de destilação subsequente ajusta os pesos restantes para restaurar a qualidade.

**A execução pode ser interrompida?**
Sim. Ao interromper com `Ctrl-C`, o estado atual é salvo em `state.json` e checkpoints de treinamento são persistidos para retomada posterior.

# Compressão por tarefa na CLI

Implementação em cinco milestones, 8/9/2026. O objetivo é aceitar novas famílias por contratos de tarefa e operadores, sem exigir que todo modelo tenha a estrutura de um decoder Qwen. Isso **não certifica suporte universal**: cada modelo ainda precisa carregar seus pesos treinados, preservar sua tarefa, converter seus operadores e passar pela avaliação nativa.

## Caminhos disponíveis

| Comando | Responsabilidade |
| --- | --- |
| `inspect MODEL` | Identifica tarefa, modalidades e estrutura sem carregar pesos; registra recursos locais e perfil de destino. |
| `fetch MODEL -o DIR` | Baixa pesos safetensors e assets selecionados, com revisão fixada e retomada. Inclui `vocab.txt` e assets SentencePiece. |
| `compress LOCAL --data FILE -o DIR` | Compara receitas sem exigir poda; avalia tarefa, converte e executa cada candidato elegível no Core AI. |
| `predict BUNDLE --inputs INPUT.npz -o OUTPUT.npz` | Executa a interface estática exportada com tensores preparados. |
| `score-answers FILE.jsonl` | Mede correspondência com respostas aceitas e repetição periódica em gerações já produzidas. |
| `run MODEL --recovery full\|lora` | Caminho causal de poda/reconstrução e recuperação, com exportação que inclui prefill e KV cache. |

`compress` usa processos separados por candidato e execução sequencial. `--task auto` infere a tarefa dos metadados; configurações ambíguas exigem declaração explícita. Os adaptadores HF embutidos são `causal-lm`, `masked-lm` e `classification` textual. Recusam pesos ausentes/incompatíveis e carregadores pré-quantizados não suportados, em vez de inicializar uma cabeça e apresentá-la como treinada.

## Dados e aprovação

Para linguagem causal, cada linha é um documento com split explícito:

```json
{"split":"calibration","text":"Texto representativo do domínio."}
{"split":"validation","text":"Outro documento, para escolher a receita."}
{"split":"test","text":"Documento reservado para testar o selecionado."}
```

Essas três linhas ilustram o formato, não um volume suficiente para julgar qualidade. A CLI exige splits não vazios e recusa entradas tokenizadas idênticas entre eles. Próximas duplicatas e contaminação semântica continuam responsabilidade da preparação dos dados.

Conversas usam `messages` e, opcionalmente, `template_kwargs`. A avaliação causal de conversas considera somente respostas delimitadas pelo template. Templates que reescrevem prefixos ou não permitem limites inequívocos exigem um adaptador específico. Não são substituídos por um template genérico.

Classificação textual exige `label` inteiro. MLM exige `label_ids` com exatamente `--seq-len` posições, usando `-100` onde não se avalia. Os IDs são específicos do tokenizer. Truncamentos são contados nos metadados; para avaliar outros comprimentos, execute outro contrato.

Métricas `loss:` diminuem com a melhora; métricas `score:` aumentam. O padrão permite até 5% de aumento da **entropia cruzada**, queda absoluta de 0,02 em scores e RMSE relativo máximo de 0,05. Entropia cruzada e perplexidade são medidas distintas: os 5% do comando `compress` não são 5% de PPL. O erro relativo usa o pior grupo de saída, impedindo que uma saída grande esconda a corrupção de outra pequena.

Declare também qualidade absoluta quando ela for requisito do uso, por exemplo em `policy.json`:

```json
{
  "max_relative_rmse": 0.05,
  "max_loss_ratio": 1.05,
  "max_score_drop": 0.02,
  "min_scores": {"score:accuracy": 0.9}
}
```

```bash
aguardente compress modelos/classificador --task classification --data dados.jsonl \
  -o run/classificador --quality-policy policy.json --recipes w8,w4-block32
```

Uma referência com acurácia zero não satisfaz esse piso, mesmo se a conversão a reproduzir perfeitamente. Um relatório `accepted` se refere aos critérios e amostras declarados; sem um benchmark factual, ele não comprova factualidade.

## Receitas e seleção

| Receita | Operação efetiva |
| --- | --- |
| `none` | Exporta o modelo FP32 sem compressão de pesos pela receita. |
| `w8` | Quantização de pesos de 8 bits. |
| `w4-block32`, `w4-block128` | Quantização de pesos em blocos de 32 ou 128. |
| `mixed4-8` | W4 com `q_proj`, `k_proj`, `lm_head`, `classifier` e `score` explicitamente preservados em W8 quando presentes. |
| `palette4`, `palette6` | Paletização K-means do SDK com tabelas de 4 ou 6 bits. |

O padrão de mistura é uma heurística declarada, não uma busca de sensibilidade aprendida. Dimensões incompatíveis com blocos W4 usam W8 explicitamente. Vetores de normalização permanecem em sua precisão original; grupos indivisíveis de paletização também são preservados. `overrides` registra essas decisões. O rótulo da receita não implica que todo tensor tenha aquela precisão nem garante tamanho menor.

As receitas atuais comprimem **pesos**. Passar dados pelo grafo não as transforma em AWQ/GPTQ ou reconstrução orientada por ativações. Esses métodos permanecem fora desta entrega.

Cada candidato passa por comparação PyTorch, exportação verdadeira e execução Swift/Core AI. Falhas de compilação ou runtime ficam registradas como `failed`, com log próprio. Uma receita independente pode prosseguir com os mesmos critérios. Nenhum erro torna um candidato aprovado.

Entre os aprovados, o programa calcula a fronteira dos candidatos que não são superados simultaneamente em tamanho, fidelidade, métricas de tarefa, RSS e mediana de chamadas. `--select size|latency|fidelity` escolhe um objetivo nessa fronteira; o padrão é `size`. Poucas chamadas não constituem benchmark estável de latência.

Somente o escolhido é executado sobre o teste final. Se reprovar, a execução reprova; o teste não é usado para escolher outro. Os exemplos de integração distribuídos abaixo já foram observados durante desenvolvimento e não servem como benchmark independente futuro.

## Recursos e retomada

`--target-ram-gib 8 --target-reserve-gib 4` registra o destino, independentemente da máquina de preparação. O perfil limita o RSS observado do processo nativo; `--max-asset-mib` limita o arquivo. **Nenhum deles certifica toda a memória unificada**: alocações GPU, especialização e temporários ainda precisam ser medidos no destino. Por isso `runtime_memory_verified` permanece `false`.

O caminho HF de `compress` prepara o modelo inteiro em FP32 na CPU e estima espaço para duas cópias dos pesos antes de iniciar. Não permite carregar qualquer modelo maior que a RAM. O caminho LoRA reduz o custo de gradientes e AdamW, mas mantém a base inteira residente. Dados e saídas de referência também ocupam RAM/disco; entradas grandes por adaptador exigem dimensionamento próprio.

O diretório contém `spec.json`, `baseline/`, diretórios de receita, logs e `selection.json`. A identidade inclui conteúdo dos pesos/dados, arquivo do adaptador, implementação e versões dos pacotes principais. Referências, relatórios e bundle são verificados por hashes. Alterar a configuração ou os ingredientes exige outro diretório, preservando a execução anterior. Retomar repete o teste nativo final; não repete candidatos concluídos íntegros. Receitas que falharam são tentadas novamente.

O arquivo de adaptador é código local escolhido explicitamente pelo usuário. Seus imports externos e recursos adicionais precisam ser versionados pelo autor; a identidade automática cobre o arquivo declarado, não todas as dependências transitivas de um plugin.

## Recuperação LoRA e respostas

```bash
aguardente run modelos/decoder -o run/recuperacao --target-params 120e6 \
  --recovery lora --lora-rank 8 --lora-alpha 16 \
  --calib-file conversas.jsonl --assistant-only --batch-size 1 --seq-len 256
```

O arquivo de conversas de `run` precisa de pelo menos 20 documentos distintos para separar treino, validação e teste. `--assistant-only` exige `messages`: preserva máscaras nos documentos, packing, shards de logits, loss e avaliação de recuperação. Prompts permanecem como contexto e não recebem perda como alvos; tokens de término reais continuam supervisionados mesmo quando EOS também é o ID de padding.

LoRA adapta lineares elegíveis com base congelada, A/B e momentos em FP32. No MPS, a base usa BF16. Pesos compartilhados e a saída de embeddings ficam protegidos por padrão. A fusão calcula deltas em blocos de linhas e recusa pesos não finitos antes de modificar a base. Depois verifica a perplexidade antes/depois da fusão e salva um checkpoint HF comum, sem exigir adaptadores no dispositivo de destino.

LoRA **não reduz o tamanho da base sozinho**. Neste comando, a economia estrutural vem da poda, quando solicitada; a economia de representação vem da compressão posterior. O modo não implementa QLoRA nem torna o pipeline de recuperação causal automaticamente compatível com visão, MoE ou qualquer projeção fundida.

Para avaliar gerações já coletadas:

```json
{"answer":"Mercúrio","expected":["Mercúrio","Mercury"]}
```

```bash
aguardente score-answers respostas.jsonl --min-exact-match 0.9 --max-repetition 0.02
```

O comando retorna código 1 quando os limites falham. Correspondência exata normalizada é apropriada a respostas curtas; não julga explicações livres ou prova factualidade geral. A referência deve ser verificada, e não apenas copiada do teacher.

## Adaptador de outra tarefa

`--adapter examples/classification_task.py:build` exemplifica um classificador de vetores com pesos já treinados. A factory recebe `(model_ref, data_path)` e retorna `aguardente.tasks.ModelTask` com:

- `model`: módulo PyTorch novo, com pesos treinados carregados integralmente.
- `input_names`, `output_names`: nomes únicos para tensores.
- `samples`: lotes reais nos três splits, como tuplas de tensores de formas estáticas.
- `score(outputs, split)`: métricas da tarefa comparadas com referências do respectivo split.
- `description`, `preprocessing`: contrato e metadados; um tokenizer fornecido é salvo no bundle.

```bash
aguardente compress modelos/classificador --adapter examples/classification_task.py:build \
  --task custom --data vetores.jsonl -o run/vetores --recipes w8,none
```

A ponte atual aceita entradas `float32`, `float16`, `int32` e saídas `float32`/`float16`; batch e formas são fixos. Recusa mutações ocultas de buffers/entradas. Um estado recorrente pode ser explicitado como entrada/saída por um adaptador, mas seu ciclo temporal e avaliação precisam ser implementados pelo autor. Não há geração genérica otimizada com KV cache nesse caminho. O caminho causal de `run` mantém sua ponte específica de prefill/decode.

Tarefas de imagem, áudio, embeddings, encoder-decoder ou difusão precisam preservar processamento, estados e métricas apropriadas. O exemplo de vetores demonstra a extensão; não é evidência de que todas essas famílias já foram convertidas.

O bundle contém `model.aimodel`, `interface.json` e assets fornecidos pelo adaptador. Para inferência, prepare um NPZ com os nomes, dtypes e formas exatos de `interface.json`:

```bash
aguardente predict run/vetores/w8/bundle --inputs entradas.npz -o saidas.npz
```

Esse comando não carrega o teacher. O runtime usa Core AI com preferência GPU e exige macOS 27/Xcode compatíveis. Leve o bundle completo ao destino, não apenas o arquivo de pesos.

## Ensaios realizados no M1 Air de 8 GiB

Protocolos de integração com três exemplos de calibração, dois de validação e dois de teste, sequência 32. São amostras pequenas e conhecidas, inadequadas para certificar qualidade geral.

```bash
aguardente fetch HuggingFaceTB/SmolLM2-135M -o modelos/smollm
aguardente compress modelos/smollm --data docs/examples/language-smoke.jsonl \
  -o run/smollm --recipes w8 --seq-len 32

aguardente fetch google/bert_uncased_L-2_H-128_A-2 -o modelos/bert
aguardente compress modelos/bert --task masked-lm --data docs/examples/bert-masked-smoke.jsonl \
  -o run/bert --recipes w8,w4-block32,palette6 --seq-len 32
```

| Ensaio | Resultado nativo no teste |
| --- | --- |
| SmolLM2, W8 estático | Asset 136.137.692 bytes; RMSE relativo 0,03197; entropia cruzada 3,54859. |
| BERT, W8 estático | Asset 4.696.307 bytes; RMSE relativo 0,02001; entropia cruzada 4,50193; acurácia 0/2, igual à referência. |
| BERT, W4 bloco 32 | Reprovado em validação por erro numérico e entropia cruzada, antes da exportação. |
| BERT, paletização 6 bits | Aprovada em validação, mas asset de 16.169.633 bytes; W8 venceu o objetivo de tamanho. |
| SmolLM2, LoRA rank 4 e exportação causal | Quatro passos, 1.221.120 parâmetros treináveis; PPL de recuperação permaneceu 17,95; razão Core AI/PyTorch no teste de runtime 1,00103. |

O checkpoint BERT veio da revisão `30b0a37ccaaa32f332884b96992754e246e48c5f`. Sua cabeça treinada de MLM foi usada explicitamente; os pesos de pooling/NSP não usados pela tarefa constam nos metadados. O SmolLM2 veio do snapshot local da auditoria anterior, sem manifesto remoto de commit, e foi fixado pelos hashes de conteúdo desta execução. Uma nova chamada de `fetch` pode resolver outra revisão; confira o manifesto antes de comparar.

No teste final estático, o RSS de host foi aproximadamente 1,05 GiB no SmolLM2 e 72,3 MiB no BERT. Houve RSS maior durante a primeira especialização do SmolLM2. Isso evidencia que tamanho do arquivo e custo do runtime são diferentes; os valores não medem toda a memória GPU nem uma sessão prolongada.

O ensaio LoRA demonstra treinamento, fusão e execução, **sem demonstrar melhoria de linguagem**. As correções de máscara tornam o objetivo de treinamento adequado para respostas, mas ainda precisam de experimentos representativos para medir sua contribuição. Melhorar factualidade continua exigindo referência adequada, dados corretos e avaliação independente.

O adaptador local de classificação também foi executado de ponta a ponta, usando uma rede de 4 entradas e 16 unidades ocultas treinada sobre uma função sintética de classificação. Tanto W8 quanto FP32 passaram; o teste reservado acertou 15/15. O programa escolheu `none`: nesse modelo minúsculo, os metadados da quantização produziram 5.296 bytes contra 3.686 bytes sem quantização. É um teste funcional da extensão e da seleção por tamanho, não um benchmark de classificação real.

A suíte completa passou com 668 testes e dependências de pipeline presentes, incluindo interrupção/retomada com LoRA e máscaras de resposta. O wheel foi construído e conferido quanto à inclusão das pontes Swift; a CLI instalada também executou `predict` sobre o BERT. A retomada real do SmolLM2 reutilizou o candidato verificado e repetiu somente o teste final.

## O que ainda falta

Preparação por blocos para pesos maiores que a RAM; quantização orientada por ativações e reconstrução dos pesos quantizados; QLoRA/MLX; alocação aprendida de bits/ranks; poda/fusão de experts; estados genéricos eficientes; benchmarks amplos por tarefa; medição completa de memória unificada e contextos longos; execução comprovada de famílias multimodais completas.

Preparar em uma máquina com mais RAM e cálculo pode permitir melhores dados, calibração, teachers e recuperação, produzindo um artefato melhor dentro do orçamento do Air. Converter os mesmos pesos com a mesma receita em um chip mais novo não acrescenta conhecimento. A qualidade e os recursos do pacote final precisam ser avaliados no próprio M1. Não houve ensaio com um M5 Pro nesta entrega.

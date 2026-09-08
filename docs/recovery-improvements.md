# Recuperação e validação — 7 de setembro de 2026

Esta revisão continua a [auditoria inicial](pipeline-audit.md). **O pipeline completo passou com SmolLM2-135M reduzido para 121,24 M parâmetros e convertido em int8; os ensaios de 114,61 M foram reprovados.** A recuperação ficou mais eficaz e a conversão preservou numericamente o student, mas a geração ainda contém repetição e erros factuais. Destilar qualquer arquitetura, para qualquer tamanho, preservando capacidade, continua fora do que este código ou estes experimentos demonstram.

## Correções aplicadas

| Etapa | Problema | Comportamento atual |
|---|---|---|
| Download | Arquivo do mesmo tamanho podia estar corrompido; `main` podia mudar entre retomadas; a árvore podia ser paginada. | Commit e manifesto de arquivos fixados em `.aguardente-source.json`; paginação da árvore; SHA-256 dos pesos LFS conferido inclusive após uma etapa marcada como concluída. |
| Planejamento | Um `lm_head` serializado era interpretado como peso independente mesmo quando o config declarava compartilhamento. | O config determina compartilhamento; a contagem em arquivo permanece separada da contagem de parâmetros do decoder. |
| Recursos | Falha ao medir RAM podia aparecer como máquina de 0 GB, com sugestão de ignorar o teto. | Orçamento recusa RAM desconhecida com diagnóstico da medição; a recomendação para falta de RAM é reduzir parâmetros ou usar outra máquina. |
| Entrada de student externo | Igualdade de `vocab_size` permitia comparar tokens de significados diferentes. | Exige equivalência da definição do tokenizer fast, incluindo normalização, segmentação, IDs especiais e chat template. |
| Dados | Truncamento por documento, padding excessivo e descarte do lote final desperdiçavam exemplos. | Packing com EOS e sobreposição de um token entre janelas; máscaras preservam EOS real e o lote residual. |
| Separação de dados | Dados e ordem não ficavam fixados como parte do experimento. | Snapshot dos documentos de treino/validação/teste, deduplicação exata, hashes e semente; datasets/TXT/JSONL customizados. |
| Poda | Apenas remover colunas perdia contribuições que canais correlacionados poderiam representar. | Ajuste ridge das projeções de saída de MLP/atenção; comparação com poda simples em validação. |
| Destilação | Agregar toda a cauda numa categoria removia sua estrutura, embora ali estivesse a maior parte da probabilidade. | Top-k exato mais amostras condicionais da cauda para estimar KL do vocabulário inteiro. |
| Gradiente | Microbatches com quantidades diferentes de tokens válidos recebiam o mesmo peso. | Acumula soma de perdas e normaliza pelos tokens válidos da janela completa. |
| Checkpoint | Limiar de paciência podia deixar de salvar uma melhora pequena; treino ruim descartava o student inicial. | Salva todo mínimo real, incluindo o checkpoint inicial, independentemente da regra de parada. |
| Parada antecipada | A v5 parou por plateau após consumir somente metade da primeira época. | Plateau só pode encerrar depois de uma passagem completa pelo corpus; limites de tempo e interrupções continuam respeitados. A política participa da assinatura de recuperação/exportação. |
| Retomada | Dropout/ordem dos lotes e uma janela incompleta podiam alterar o resultado. | Persiste RNG e estado do otimizador; repete a janela não confirmada; Ctrl-C aguarda concluir um passo AdamW já iniciado. |
| Precisão e memória dos checkpoints | Student FP32 era recarregado em BF16 antes de voltar a FP32; embeddings compartilhados podiam ser copiados duas vezes para CPU. | Carregamento da recuperação diretamente em FP32 e preservação de aliases na cópia do estado para CPU. |
| Concorrência | `--restart` removia arquivos antes do lock; lock podia expirar durante treino vivo. | Limpeza dentro do lock; um processo local vivo mantém a proteção independentemente da duração. |
| Aprovação | Treino podia terminar e exportar um modelo de qualidade muito inferior. | Teste separado exige razão de perplexidade até 1,2 por padrão; reprovação preserva pesos, sem exportação aprovada. |
| Compressão | Int4 era presumido adequado, embora degradado no teste real anterior. | `auto` seleciona entre int4, int8 e sem quantização em validação; testa o vencedor separadamente. |
| Artefatos | Mais de um bundle podia resultar em escolha por data de modificação. | Diretórios isolados por assinatura/candidato; ambiguidade é erro; estado registra o asset selecionado. |
| Configuração de exportação | Incompatibilidade podia ser descoberta depois de treinar; receitas de ativações podiam calibrar com dados alheios à execução. | Verificação de arquitetura/contexto/receita antes do treino; receitas que exigem ativações podem usar somente o split de treino do snapshot. |
| Runtime | Estados KV acumulavam; subprocesso sem saída podia ignorar timeout. | Liberação explícita de estados, timeout durante a leitura e limpeza dos processos; prefill em blocos seguido de decode. |
| Saída e status | O launcher de desenvolvimento ignorava o código retornado pela CLI; progresso sem TTY podia ficar invisível. | `scripts/dev_cli.py` propaga o código de saída; logs sem TTY recebem atualizações periódicas; status reconhece reprovação de qualidade. |

A inferência Core AI continua usando a ponte Swift descrita na auditoria inicial. Não foi corrigido o compilador da Apple: foi corrigida a integração com as opções públicas do SDK instalado. Crash nativo interrompe a seleção automática; não vira justificativa para escolher outra receita ou omitir validação.

## O que mudou na função de destilação

Na rodada v2, uma amostragem de um a cada 16 shards mediu massa média de **25,51% no top-128**, com temperatura 2. Agregar os 74,49% restantes preservava a massa da cauda, mas perdia a distinção entre seus tokens. A revisão usa:

```text
KL_est = soma_topk p(i) log(p(i)/q(i))
         + p(cauda) * média_amostras log(p(z)/q(z))
z ~ p(. | fora do top-k), com reposição
loss = alpha * T² * KL_est + (1-alpha) * CE_causal
```

O gradiente coincide em esperança com o da KL completa. O cache mantém uma amostra finita fixa nas épocas; portanto há erro de Monte Carlo, que aumentar o número de amostras reduz, com custo de disco e leitura. Não é KL completa exata, e não é uma prova de recuperação de capacidade. Um teste com distribuição pequena e 100 mil amostras compara seu gradiente com a KL completa. O código não trunca estimativas negativas de KL em zero, pois isso enviesaria o gradiente.

A representação custa `8 × (K + N) + 4` bytes por posição, fora tokens/máscara/serialização. O manifesto registra a massa efetivamente coberta pelo top-k. A API ainda lê shards K+1 sem amostras; o pipeline da CLI gera amostras da cauda com quantidade positiva.

## Experimentos reais

Máquina: Apple M1 de 8 GB, macOS 27 beta, Python 3.12, PyTorch 2.9.0, Transformers 5.12.1, Core AI do ambiente registrado na auditoria inicial. Teacher: **SmolLM2-135M base**, não instruct. A primeira tabela reúne as rodadas até v5, com 134.515.008 → 114.608.448 parâmetros, cortando somente MLP de 1.536 para 1.152 canais. As 30 camadas e a atenção permanecem. O ensaio v6, separado abaixo, usa corte menor.

Protocolo de qualidade: primeiros 20.000 caracteres do WikiText-2 test, janela 512 e stride 256. Teacher inferido em BF16; student avaliado em FP32. Menor perplexidade é melhor. Não equivale a benchmark de conversação, código, raciocínio ou português.

| Rodada | Transições distintas de treino | PPL após poda/reconstrução | PPL após treino | PPL student/teacher |
|---|---:|---:|---:|---:|
| Auditoria inicial | 9.435 | 32,113 | 30,304 | 1,688 |
| v2: packing + reconstrução + seleção de checkpoint | 65.280 | 26,447 | 22,890 | 1,275 — reprovada |
| v3: cauda amostrada e mais dados | 261.120 | 26,100 | 22,218 | 1,237 — reprovada |
| v4: continuação da v3 | 522.240 | 22,218 (student da v3) | 21,658 | 1,206 — reprovada |
| v5: continuação da v4 | 130.560 consumidas de 261.120 previstas | 21,658 (student da v4) | 21,584 | 1,202 — reprovada |

Teacher: **17,954**. Na v2, a reconstrução reduziu a perplexidade de validação de **20,299 para 18,037**. O treino terminou por plateau após 176 passos, com 179.520 transições vistas, e restaurou o mínimo de validação de 15,359. Recuperou 41,9% da diferença de perplexidade entre seu ponto de partida podado e o teacher; isso não significa 41,9% de capacidade recuperada. O student ficou 24,5% melhor em perplexidade que o resultado da auditoria inicial, mas ainda 27,5% pior que o teacher e **não seguiu para exportação**. O limite de 20% não foi alterado para aprová-lo.

Artefatos v2: [log](../.local/recovery-v2-smollm.log), [qualidade](../.local/recovery-v2/smollm/quality.json), [reconstrução](../.local/recovery-v2/smollm/reconstruction.json), [treino](../.local/recovery-v2/smollm/ckpt/recovery.json).

A rodada v3 acrescentou amostragem da cauda e mais dados: 1.024 janelas de 256 tokens, **261.120 transições distintas**, top-192 e 256 amostras da cauda por posição. O top-k cobriu 28,21% da probabilidade média. Na preparação, reconstrução melhorou a validação de 20,742 para 18,065. Foram 512 passos, duas épocas e 522.240 transições processadas, em 2.158,45 segundos de recuperação. O mínimo de validação foi 14,771 no passo final. O teste chegou a **22,218**, recuperando 47,7% da diferença de perplexidade entre poda/reconstrução e teacher, mas ainda **23,75% acima do teacher**. A exportação foi bloqueada; não se elevou o limite para aprovar o resultado.

Artefatos v3: [log](../.local/recovery-v3-smollm.log), [qualidade](../.local/recovery-v3/smollm/quality.json) e [recuperação](../.local/recovery-v3/smollm/ckpt/recovery.json). Uma amostra de `vm.swapusage` durante esse treino mostrou aproximadamente 6,5 GB de swap total do sistema. Isso não mede memória exclusiva do processo nem pico, mas evidencia pressão de memória no M1 de 8 GB. Os testes do código não equivalem a ausência de paginação em uso real.

Nas rodadas v2/v3, `quality.json` e a mensagem da CLI registraram a reprovação e impediram a exportação, mas o launcher de desenvolvimento usado no ensaio devolvia zero ao shell por ignorar o retorno de `main()`. Esse defeito do launcher foi corrigido; não se deve usar seu código de saída antigo como evidência de aprovação.

Comando da v3, usando o Python com as dependências completas e o código deste checkout:

```bash
PIPELINE_PY=/Users/joaocosta/.local/share/uv/tools/aguardente/bin/python
"$PIPELINE_PY" scripts/dev_cli.py run .local/audit-smollm/teacher \
  -o .local/recovery-v3/smollm --target-params 120e6 --effort high \
  --batch-size 1 --seq-len 256 --logit-batches 512 --calib-batches 64 \
  --epochs 2 --grad-accum 4 --lr 0.0001 --max-context-length 512 \
  --measure --device mps
```

O teacher já estava local. O download também foi repetido pela CLI em `.local/recovery-v3/fetched`, com verificação de checksum concluída:

- Commit: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- SHA-256 de `model.safetensors`: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
- [Log do download](../.local/recovery-v3-fetch.log) e [proveniência fixada](../.local/recovery-v3/fetched/.aguardente-source.json).

Use outro diretório para um novo experimento. A semente, o snapshot e o checkpoint permitem retomar o mesmo treino; versões de bibliotecas e dispositivos diferentes não têm promessa de igualdade bit a bit.

### Continuação a partir do student recuperado

A v4 usa o student da v3 como entrada explícita de `--student`, com o mesmo teacher, contagem de parâmetros e limite de qualidade. Não pula destilação: inicia um novo AdamW com LR menor e mais dados. A configuração é uma época, 2.048 janelas de 256 tokens, `--lr 0.00005` e `--seed 43`; o corpus foi ampliado. Há documentos compartilhados entre as rodadas; não se afirma que todas as transições sejam inéditas em relação à v3. Os splits de validação e teste permanecem iguais para comparação, e não constituem um teste cego intocado por todo o desenvolvimento.

```bash
"$PIPELINE_PY" scripts/dev_cli.py run .local/audit-smollm/teacher \
  -o .local/recovery-v4/smollm --student .local/recovery-v3/smollm/student \
  --effort high --batch-size 1 --seq-len 256 --logit-batches 1024 \
  --calib-batches 64 --epochs 1 --grad-accum 4 --lr 0.00005 --seed 43 \
  --max-context-length 512 --device mps
```

A v4 terminou com **21,6579** no teste, razão **1,20628**, ainda acima de 1,2. Foi reprovada e não exportada. O mínimo de validação foi 14,32490 no passo 512. O processo original deixou de existir antes de finalizar; em 7/9, a CLI retomou o checkpoint do passo 450 e concluiu os 62 restantes, reutilizando os logits. O campo `seconds = 231,05` de `recovery.json` mede somente essa sessão retomada, não a rodada inteira. A CLI corrigida retornou código 1.

Artefatos v4: [log original](../.local/recovery-v4-smollm.log), [retomada](../.local/recovery-v4-resume.log), [qualidade](../.local/recovery-v4/smollm/quality.json) e [treino](../.local/recovery-v4/smollm/ckpt/recovery.json).

A v5 continua esse mesmo student de 114.608.448 parâmetros, sem alterar o limite. Usa outro embaralhamento do treino e novo AdamW, com LR menor, 1.024 janelas e uma época. Os resultados das rodadas anteriores continuam registrados como reprovações.

```bash
"$PIPELINE_PY" scripts/dev_cli.py run .local/audit-smollm/teacher \
  -o .local/recovery-v5/smollm --student .local/recovery-v4/smollm/student \
  --effort high --batch-size 1 --seq-len 256 --logit-batches 512 \
  --calib-batches 64 --epochs 1 --grad-accum 4 --lr 0.00003 --seed 44 \
  --max-context-length 512 --device mps
```

A v5 terminou por plateau no passo 128, após 130.560 transições e 461,68 segundos. O mínimo de validação caiu para 14,28265, mas a melhora relativa de 0,295% não alcançava o limiar de paciência de 0,5%. O melhor checkpoint foi corretamente salvo; ainda assim, parar antes de consumir metade do orçamento impediu avaliar o restante do corpus. Isso motivou a correção de parada antecipada descrita acima. O resultado registrado continua sendo **21,58360**, razão **1,20214**, reprovado com saída 1 e sem exportação. Não se afirma que a nova regra teria aprovado esse student sem executar o restante.

Artefatos v5: [log](../.local/recovery-v5-smollm.log), [qualidade](../.local/recovery-v5/smollm/quality.json) e [treino](../.local/recovery-v5/smollm/ckpt/recovery.json).

### Ensaio completo aprovado: v6, com corte menor

Este é outro experimento, iniciado no teacher original. O alvo explícito passou a **125 M**, resultando em **121.243.968 parâmetros** pelo alinhamento da MLP: 1.536 → 1.280 canais. É redução de 9,87% dos parâmetros, e não aprovação retroativa dos students de 114,61 M. A regra de uma época mínima antes de plateau já estava aplicada.

```bash
"$PIPELINE_PY" scripts/dev_cli.py run .local/audit-smollm/teacher \
  -o .local/recovery-v6/smollm --target-params 125e6 --effort high \
  --batch-size 1 --seq-len 256 --logit-batches 256 --calib-batches 64 \
  --epochs 2 --grad-accum 4 --lr 0.0001 --seed 42 \
  --max-context-length 512 --device mps
```

Nenhum `--skip-*`, tolerância ampliada ou supressão de falha foi usado. O download havia sido executado pela CLI e verificado separadamente, como registrado acima. `fetch` aparece dispensado porque a entrada desse comando é o diretório local já baixado; a extração é dispensada porque o teacher já é um decoder causal canônico.

A reconstrução das 30 projeções melhorou a validação de **16,676 para 15,430** e foi aceita. Foram preparados 512 shards, **130.560 transições distintas**, top-192 cobrindo 28,01% da massa de probabilidade e 256 amostras da cauda. O treino consumiu 195.840 transições em 192 passos (uma época e meia), parou por plateau e restaurou o mínimo real de validação de **13,71095** no passo 192. A recuperação registrou **683,90 segundos**.

| Etapa | PPL de teste, 4.730 tokens |
|---|---:|
| Teacher BF16 | 17,95429 |
| Student após poda/reconstrução, FP32 | 22,54395 |
| Student recuperado, FP32 | **20,22108** |

A razão student/teacher foi **1,12625**, abaixo do limite inalterado de 1,2. Recuperou 50,6% da diferença de perplexidade entre o ponto podado/reconstruído e o teacher. Isso não mede percentual de capacidades recuperadas.

A seleção de exportação executou ambos os candidatos no Core AI. Int4, com 65,43 MiB, **falhou**: sua PPL de validação aumentou 58,39%, além de exceder tolerâncias de logits e prefill em blocos. Int8 foi selecionado e passou no teste separado. Não foi necessário testar o candidato FP16 sem quantização nesta rodada.

O protocolo de runtime abaixo usa os mesmos primeiros 6.000 caracteres do test, 1.549 transições, janela 256 e stride 128; por isso seus valores diferem da tabela de recuperação. O teacher apenas quantizado é uma referência separada, sem poda ou treino, para medir a utilidade da redução.

| Modelo | Asset | PPL PyTorch FP32 | PPL Core AI | Core AI / PyTorch |
|---|---:|---:|---:|---:|
| Teacher int8 | 136,67 MiB | 19,75233 | 19,79656 | 1,00224 |
| Student v6 int8 | **123,23 MiB** | 21,25058 | **21,21258** | **0,99821** |

O student economizou **9,84% no asset** contra o teacher int8, com PPL nativa **7,15% maior** nesse recorte. Os três prompts tiveram erro RMS relativo de 2,04–3,09% contra PyTorch. Prefill em blocos de 128, seguido de decode em um prefixo de 511 tokens, teve erro RMS relativo de 2,55%. O pipeline e sua retomada concluíram com código 0; a retomada reutilizou os pesos e revalidou o runtime.

As gerações não são satisfatórias para uso geral: o student repete a ideia de aprender novas palavras, produz afirmações falsas sobre os planetas e continua fraco em português. Há problemas no próprio checkpoint PyTorch; a quantização também pode mudar trajetórias greedy mesmo passando nas tolerâncias. **Neste modelo pequeno, apenas quantizar o teacher é uma referência mais favorável: cabe folgadamente como asset e preserva mais qualidade.** Não foram medidos pico de RAM de inferência, latência sustentada ou energia para concluir vantagem de execução do student.

Evidências: [execução completa](../.local/recovery-v6-smollm.log), [retomada](../.local/recovery-v6-resume.log), [qualidade](../.local/recovery-v6/smollm/quality.json), [reconstrução](../.local/recovery-v6/smollm/reconstruction.json), [treino](../.local/recovery-v6/smollm/ckpt/recovery.json), [seleção](../.local/recovery-v6/smollm/bundle/813d8907f81cf423/selection.json), [runtime do student](../.local/recovery-v6/smollm/runtime-validation.json), [runtime do teacher](../.local/recovery-v6/teacher-runtime.json) e [asset aprovado](../.local/recovery-v6/smollm/bundle/813d8907f81cf423/8bit/student_8bit_dynamic/student_8bit_dynamic.aimodel).

## Limites e revisão por etapa

1. **Download e entradas.** Há integridade dos pesos LFS, manifesto para retomada offline e revisão do modelo fixada. A listagem segue paginação sem mudar de repositório/revisão. O dataset remoto ainda não tem revisão fixada, mas o snapshot conserva os textos exatos usados. O caminho urllib/aria2 ainda não integra autenticação do Hub; baixar previamente um modelo autorizado com o cliente HF e usar seu diretório local continua necessário para esse caso. O carregador de modelos inteiros ainda exige que o teacher caiba em memória. Não foi implementado streaming dos pesos por camada ou teacher distribuído.
2. **Planejamento.** Continua sendo uma política de redução com MLP antes de atenção e profundidade. Contagem de parâmetros não prevê qualidade; não há busca de arquitetura validada sob o mesmo orçamento. O alinhamento de 128 pode produzir um resultado menor que o teto solicitado. A memória de treinamento FP32 continua em pelo menos 16 bytes por parâmetro, além de ativações e temporários. A sondagem real do Qwen3-1.7B em 7/9 confirmou 1.720.574.976 parâmetros únicos, com 2.031.739.904 valores armazenados: seu [config declara embeddings compartilhados](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json). O piso estrutural calculado é aproximadamente 531 M, acima do teto estimado de treino de 201 M deste M1 de 8 GB. Foi executado somente o planejamento do Qwen, não download completo, treino ou inferência.
3. **Dados.** Packing e separação corrigem desperdício e contaminação exata. Um corpus genérico curto não preserva automaticamente competências especializadas. JSONL com mensagens respeita o template, mas o objetivo ainda treina todos os tokens: não há loss exclusiva das respostas do assistant, geração de dados pelo teacher ou benchmark de instruções. É necessário fornecer dados e avaliações do domínio pretendido.
4. **Poda e reconstrução.** Os pesos realmente encolhem. A compensação ridge usa correlações locais e melhora este ensaio; não recupera subespaços ou camadas inteiras arbitrariamente removidos. Cabeças e camadas foram exercitadas em testes sintéticos de Llama, Mistral, Qwen2 e Qwen3, com salvamento/recarregamento e passos KD; isso não mede qualidade de modelos pré-treinados dessas quatro famílias. A validação multimodal continua restrita ao escopo da auditoria anterior.
5. **Destilação.** O objetivo agora representa a cauda, usa precisão adequada e seleciona o mínimo real em validação. Treino maior não garante qualidade: cortes fortes podem precisar de muito mais dados, outro student ou outra arquitetura. Interrupção normal retoma o último estado consistente; queda de energia pode repetir trabalho desde o checkpoint periódico. Arquivos de logits ilegíveis são regenerados, não usados parcialmente.
6. **Compressão.** A seleção automática cobre somente três receitas, com tolerâncias explícitas. Não é busca exaustiva por camada, GPTQ/AWQ, QAT ou palettização. Preservar o student não corrige capacidades que ele já perdeu. Uma receita int4 reprovada permanece registrada como reprovada. O fornecimento do corpus a receitas que calibram ativações tem teste de dados; não foi executada uma receita real de quantização de ativações neste ensaio.
7. **Conversão e runtime.** A validação usa entradas reais, PPL, prefill e decode, com teste de cache em blocos até 1.024 tokens. O protocolo de conversão usa até 6.000 caracteres e janelas de 256, portanto suas PPLs não se comparam diretamente à tabela de recuperação. Contexto maior, pico de RAM, velocidade sustentada, consumo de energia, execução ANE e `.aimodelc` standalone ainda exigem medições próprias.
8. **Suporte universal.** A implementação cobre decoders causais densos com estrutura compatível e um exportador Apple correspondente. MoE, projeções fundidas, atenção latente, encoder-only, seq2seq e tokenizers diferentes exigem caminhos específicos. Recusar cedo evita produzir resultados inválidos, mas não equivale a implementar suporte. Não há evidência para chamar a ferramenta de universal ou prometer redução pesada com qualidade preservada num Air de 8 GB.

## Fundamentação e verificação

Packing segue o princípio de concatenar textos e dividir em blocos descrito no [guia causal do Transformers](https://huggingface.co/docs/transformers/tasks/language_modeling), aqui preservando o residual e as fronteiras causais. Reconstrução local e recuperação por destilação são técnicas distintas: o ajuste ridge implementado aqui não é uma implementação de SparseGPT ou Minitron. O [trabalho Minitron](https://arxiv.org/abs/2407.14679) fundamenta a necessidade de combinar poda com recuperação e dados adequados; seus resultados não validam automaticamente esta implementação ou seus orçamentos.

A seleção de compressão adota avaliação de fidelidade/tamanho com entradas reais, conforme a [skill oficial de exploração da Apple](https://github.com/apple/coreai-models/blob/main/skills/skills/model-compression-exploration/SKILL.md), numa busca limitada a três candidatos. A retomada foi testada em CPU com dropout e interrupção dentro da acumulação e do passo AdamW; a [documentação do PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html) descreve os limites de reprodutibilidade entre plataformas e versões.

Os testes de regressão novos cobrem gradiente da cauda, acumulação desigual, seleção de checkpoint inicial/mínimos pequenos, retomada, corrupção de shards, hashes, tokenizers diferentes, isolamento do corpus, quatro famílias causais sintéticas, embeddings compartilhados, época mínima antes de plateau, RAM desconhecida, seleção de export e rejeição de crash. Em 7/9, **617 testes passaram** com PyTorch/Transformers disponíveis ([log](../.local/audit-smollm/tests-current-final.log)). O wheel foi gerado offline e seus 35 arquivos Python/Swift foram comparados byte a byte com os fontes. Isso verifica o pacote e o código; a qualidade do modelo depende dos ensaios reais acima.

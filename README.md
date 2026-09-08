<p align="center">
  <img src="docs/media/icon.svg" width="128" alt="aguardente">
</p>

<h1 align="center">aguardente</h1>

<p align="center">
  Poda estruturada e destilação de modelos de linguagem para Apple Silicon via Core AI.
</p>

---

`aguardente` reduz o tamanho de modelos de linguagem (LLMs) por **poda estruturada** e treina os pesos restantes via **destilação**, gerando pacotes `.aimodel` para o Core AI da Apple. É um pipeline experimental: treinamento e exportação bem-sucedidos não garantem qualidade. A [auditoria com execução real no M1 de 8 GB](docs/pipeline-audit.md) documenta as correções, os resultados e as limitações.

O [plano de compressão por qualidade e dispositivo](docs/compression-plan.md) propõe quantização calibrada, recuperação com menor consumo de memória e outras estratégias, separando a máquina de preparação do destino. As etapas desse plano ainda não estão implementadas.

---

## Instalação

Requer o [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install --python 3.12 "aguardente[pipeline] @ git+https://github.com/joaaosc/aguardente"
```

> **Nota:** As dependências do Core AI exigem Python `>=3.11,<3.14`. `--python 3.12` fixa a versão usada pelo `uv` para o programa, sem tocar no Python do sistema.

Depois, confirme o restante do ambiente (Xcode, Metal Toolchain, `aria2`):

```bash
aguardente doctor
```

Se o programa já estiver instalado sem a extra `pipeline` (por exemplo, via `uv tool install aguardente`):

```bash
aguardente install
```

---

## Uso

Execução completa do pipeline:

```bash
aguardente run HuggingFaceTB/SmolLM2-135M -o run/smollm --target-params 125e6 \
  --effort high --batch-size 1 --seq-len 256 --logit-batches 256 --calib-batches 64 \
  --epochs 2 --grad-accum 4 --lr 0.0001 --max-context-length 512
```

O comando executa poda, reconstrução das projeções e destilação, com seleção do melhor checkpoint em validação. A exportação só prossegue se a perplexidade no teste ficar até 20% acima do teacher. O padrão `--compression auto` compara int4, int8 e FP16 sem quantização, seleciona a primeira receita que preserva numericamente o student em validação e verifica novamente no teste. Os limites são critérios de engenharia, não garantia de capacidade linguística. Veja as [melhorias e os experimentos de recuperação](docs/recovery-improvements.md).

Em vez do identificador `namespace/nome`, também é aceita a URL copiada do navegador — do Hugging Face diretamente, ou do GitHub, quando o autor publica lá sob o mesmo nome:

```bash
aguardente plan https://huggingface.co/Qwen/Qwen3-4B      # a URL já contém o identificador
aguardente plan https://github.com/Qwen/Qwen3-4B          # confirmado contra o Hugging Face
```

A URL do GitHub nunca é aceita por adivinhação: `owner/repo` só vira o modelo se esse identificador existir de fato no Hugging Face, conferido por uma chamada à API. Quando não existe, o comando para e mostra os repositórios do mesmo autor e os de nome parecido que a API encontrou — nunca inventa um substituto, e nunca decide sozinho qual variante baixar.

Para inspecionar o planejamento e as dimensões estimadas sem baixar pesos:

```bash
aguardente plan Qwen/Qwen3-4B
```

Consulte **[usage.md](usage.md)** para a documentação detalhada de parâmetros e opções.

---

## Pipeline

O fluxo de processamento é dividido em duas etapas conceituais:

1. **Poda e destilação:** Reduz o número de parâmetros do modelo (estrutural) e ajusta os pesos residuais para recuperar o desempenho.
2. **Compressão/Quantização e exportação:** Converte e quantiza os pesos restantes (ex.: int4) para o formato `.aimodel` via ferramentas do Core AI.

### Etapas de execução

```
fetch  →  prune  →  logits  →  recover  →  export
```

| Etapa | Descrição |
|---|---|
| **fetch** | Download retomável, revisão fixada e SHA-256 dos pesos LFS. |
| **prune** | Poda por importância e reconstrução das projeções, selecionada por perplexidade em validação. |
| **logits** | Top-k e amostras da cauda do teacher sobre documentos empacotados com EOS. |
| **recover** | Destilação FP32, melhor checkpoint em validação e limite de qualidade no teste. |
| **export** | Seleção de compressão e execução real de `.aimodel`, com comparação numérica e perplexidade. |

### Estratégia de poda

A redução segue a proporção típica de parâmetros na arquitetura do modelo (em que camadas MLP concentram a maior parte dos parâmetros):

1. **`intermediate_size` (MLP):** Redução dos neurônios intermediários, alinhada em múltiplos de 128 para compatibilidade com quantização per-block.
2. **Cabeças de atenção:** Redução preservando grupos completos de Grouped-Query Attention (GQA).
3. **Camadas:** Remoção de camadas intermediárias baseada em *block influence*, preservando sempre a primeira e a última camada.

A dimensão `hidden_size` é mantida inalterada para preservar a consistência de embeddings, projeções e normalizações.

### Modelos multimodais

Um modelo de visão e linguagem carrega, no mesmo checkpoint, um decoder causal de texto, uma torre de visão e um projetor entre os dois. O aguardente extrai o decoder e descarta o resto: **o modelo convertido não enxerga imagens.** A preservação das capacidades de texto precisa ser avaliada para cada decoder extraído.

A extração acontece entre o download e a poda, e o restante do pipeline continua vendo um decoder causal comum.

```bash
aguardente plan deepseek-ai/deepseek-vl-7b-chat   # mostra o que será descartado
aguardente extract deepseek-ai/deepseek-vl-7b-chat -o run/dsvl   # só extrai
aguardente run Qwen/Qwen2.5-VL-3B-Instruct -o run/qwen --target-params 1.0e9
```

O prefixo do decoder não vem de uma tabela por arquitetura: é descoberto pela estrutura dos tensores, procurando o grupo de camadas cujos nomes formam um bloco transformer denso completo. Famílias conferidas contra os checkpoints publicados:

| Família | Prefixo do decoder | Decoder |
|---|---|---|
| DeepSeek-VL | `language_model.model.` | LLaMA |
| Qwen2-VL, Qwen2.5-VL | `model.` | Qwen2 |
| Qwen3-VL | `model.language_model.` | Qwen3 |
| Gemma 3 | `language_model.model.` | Gemma 3 |
| LLaVA, LLaVA-NeXT | `language_model.model.` | LLaMA, Mistral |
| SmolVLM, SmolVLM2 | `model.text_model.` | LLaMA |
| InternVL3 (porte `-hf`) | `language_model.model.` | Qwen2 |

Não são suportados, e a recusa diz o motivo: decoders MoE, atenção latente (DeepSeek-VL2), camadas de cross-attention intercaladas (Llama 3.2 Vision), e checkpoints com projeções fundidas — `qkv_proj` do Phi-3, `query_key_value` do Falcon, `attention.wqkv` do InternLM2 —, que a poda estruturada não sabe fatiar.

O `plan` também confere se o exportador da Apple aceita a arquitetura de destino, antes de qualquer download.

### Esforço da destilação

`--effort` define **quanto trabalho** se investe para recuperar a qualidade perdida na poda. É um eixo ortogonal ao alvo: `--target-params` decide o tamanho do resultado, `--effort` decide o cuidado com que se chega nele. Um alvo agressivo com esforço baixo é o caminho mais curto para um modelo pequeno e ruim.

| Nível | | Épocas | top-k | Amostras da cauda | Sequência | Logits em disco | Custo |
|---|---|---:|---:|---:|---:|---:|---:|
| `low` | ▁··· rápido | 1 | 64 | 64 | 256 | 32,1 MiB | 0,07× |
| `medium` | ▁▃·· equilibrado | 2 | 128 | 128 | 512 | 513 MiB | 1,00× |
| `high` | ▁▃▅· cuidadoso | 3 | 192 | 256 | 768 | 2.691 MiB | 4,26× |
| `max` | ▁▃▅▇ exaustivo | 4 | 256 | 512 | 1.024 | 12.296 MiB | 14,74× |

O custo é um índice de trabalho relativo ao nível `medium`, calculado pelos tokens previstos em cada etapa, não uma medição de tempo. O disco estima valores FP32, índices int32 e normalizadores antes do overhead de serialização. O número de exemplos usa lote de referência 2 e não cresce com o lote físico. Nenhum preset garante recuperação de qualidade.

Cada nível ajusta lotes de calibração, comprimento de sequência, lotes e profundidade dos logits do teacher, épocas, taxa de aprendizado, peso da destilação e acumulação de gradiente. `--batch-size` fica de fora de propósito: é restrição de memória da máquina, não escolha de qualidade.

Uma opção informada na linha de comando sempre vence o preset — `--effort max --epochs 1` usa tudo do nível exaustivo, com uma época só, e o painel diz o que foi sobrescrito.

```bash
aguardente effort          # compara os níveis, sem tocar em nenhum modelo
```

---

## Resultado medido e dimensionamento

Ensaio completo do SmolLM2-135M no M1 de 8 GB, em 7/9/2026, com a configuração acima:

|  | Original | Podado | Recuperado |
|---|---|---|---|
| Parâmetros | 134,52 M | 121,24 M | 121,24 M |
| Perplexidade¹ | 17,954 | 22,544 | 20,221 |

¹ WikiText test, primeiros 20.000 caracteres, janela 512 e stride 256. A recuperação devolveu 50,6% da diferença de perplexidade entre poda/reconstrução e teacher. O `.aimodel` int8 tem **123,23 MiB** e passou nos testes de logits, perplexidade e KV cache. Int4 foi reprovado. Os ensaios com corte maior, para 114,61 M, ficaram acima do limite de qualidade.

Isso ainda não demonstra qualidade geral de linguagem: houve repetição e erros factuais nas gerações. O teacher apenas quantizado em int8 ocupa 136,67 MiB e teve perplexidade melhor; neste modelo já pequeno, a economia de 9,8% no asset não justifica por si só destilar. Resultados, protocolos separados e revisão por etapa estão no [relatório de recuperação](docs/recovery-improvements.md); a [auditoria inicial](docs/pipeline-audit.md) preserva as medições anteriores.

A recuperação usa pesos, gradientes e dois momentos do AdamW em FP32: **16 bytes por parâmetro**, além de ativações e temporários. Com as margens atuais, o teto estático é aproximadamente 201 M parâmetros em 8 GiB de RAM e 906 M em 24 GiB; não é garantia de ausência de pressão de memória. O teacher inteiro também precisa caber antes da poda. Preparar modelos grandes pode exigir uma máquina maior que a de execução final.

A avaliação de recuperação e de runtime é obrigatória no pipeline completo; `--measure` permanece por compatibilidade. `--max-ppl-ratio` explicita o limite student/teacher (padrão 1,2). `quality.json` registra o resultado, e um student reprovado fica salvo para análise, sem seguir para exportação. A conversão admite até 5% de aumento adicional de perplexidade contra o student FP32, além dos testes de logits, prefill e KV cache.

---

## Requisitos

| Item | Requisito |
|---|---|
| Sistema | macOS 27+ com Apple Silicon |
| Xcode | Xcode 27+ com Metal Toolchain |
| Memória | Ensaio de 135 M validado em 8 GB; use `plan` para os limites estimados de teacher e student |
| Armazenamento | Depende dos pesos, logits, checkpoints e bundles; o pipeline estima o espaço necessário |

Execute `aguardente doctor` para validar o ambiente.

---

## Testes

```bash
uv run --with pytest --with torch --with transformers pytest
```

---

## Problemas Conhecidos

Levantamento do estado atual do projeto. Itens ~~tachados com ✅~~ já foram corrigidos ou mitigados; os demais permanecem em aberto, e os marcados com **[❗️prioridade❗️]** podem causar perda de trabalho ou resultado silenciosamente errado.

### Em aberto

- O orçamento estático de treino cobre 16 bytes por parâmetro. O lote automático inclui uma estimativa de ativações, mas nem ela nem a margem `train_fraction = 0,75` foram calibradas contra picos reais de memória.
- A qualidade depende do modelo, do corte e dos dados. Tokenizer externo, seleção do checkpoint, packing e quantização foram reforçados; os [novos experimentos](docs/recovery-improvements.md) distinguem melhoria medida de generalização ainda não demonstrada. A validação de cache cobre até 1.024 tokens, sem comprovar contextos maiores.
- O macOS não falha imediatamente sob pressão de memória, ele pagina. Uma execução acima do orçamento tende a ficar ordens de grandeza mais lenta em vez de abortar, e o sintoma é difícil de atribuir ao alvo escolhido.
- Um download interrompido de `xcodebuild -downloadComponent MetalToolchain` pode deixar `xcrun --find coreai-build` bem-sucedido com o componente incompleto. O diagnóstico aprovaria o ambiente e a falha só apareceria na compilação.
- A verificação de versão do Xcode aceita qualquer `27.x`, incluindo betas em que o Metal Toolchain ainda não está disponível. Na prática a verificação de `coreai-build` cobre o caso, mas o diagnóstico de versão sozinho não distingue.
- Os pesos LFS têm SHA-256 conferido; arquivos pequenos sem digest publicado continuam verificados por tamanho. A revisão do modelo é fixada, mas não há pin da revisão remota do dataset; os documentos efetivamente usados são preservados com hash.
- `probe.py` rejeita arquiteturas explicitamente não causais, mas sua cobertura de configurações desconhecidas continua limitada. Um config inesperado pode falhar tarde, já dentro da execução.
- Não foi executada uma conversão completa pela interface gráfica. O caminho CLI → NDJSON → interface está coberto por testes dos dois lados, mas nenhuma execução real de ponta a ponta foi observada.

### Corrigidos

- ~~A redução da perda de destilação (`F.kl_div` com `batchmean`) em tensores 3D dividia apenas pelo tamanho do lote, mantendo a soma sobre a dimensão temporal e multiplicando a perda de destilação pelo comprimento de sequência (`seq_len`). A perda de destilação superava a de entropia cruzada por centenas de vezes e o gradiente quadruplicava entre níveis de esforço.~~ ✅ normalização por token ativo com alinhamento causal estrito
- ~~Tokens de preenchimento (`pad_tokens`) não eram mascarados: a máscara de atenção era descartada ao gravar os shards do professor, o aluno rodava sem máscara e a perda de destilação e de entropia cruzada treinavam o modelo a prever e repetir tokens de preenchimento.~~ ✅ preservação da máscara nos shards, repasse ao aluno e exclusão de tokens mascarados no cálculo de perdas
- ~~O melhor checkpoint salvo por ganho de perplexidade (`best.pt`) nunca era recarregado no modelo: a recuperação concluía sempre com os pesos finais da última época, descartando o melhor modelo encontrado.~~ ✅ recarregamento atômico do melhor checkpoint antes da finalização
- ~~Lotes residuais em épocas com número de shards indivisível por `grad_accum` acumulavam gradientes sem nunca disparar o passo do otimizador.~~ ✅ passo residual ao término de cada época
- ~~A taxa de aprendizado decaía por cosseno até zero absoluto, tornando os passos finais inoperantes.~~ ✅ piso mínimo suave de 10% da taxa inicial
- ~~Logits com valores não finitos ou perdas divergentes (NaN/Inf) propagavam silenciosamente corrompendo os pesos do modelo.~~ ✅ validação antes do topk e antes do backward pass com `AguardenteError`
- ~~A janela deslizante de perplexidade subestimava a contagem de tokens em janelas subsequentes por subtrair um token a mais (`target_len - 1`).~~ ✅ contagem exata dos tokens não ignorados por janela
- ~~Reutilizar um diretório de execução com outro modelo ou outro alvo era silenciosamente ignorado: `RunState.load_or_create` dava precedência ao valor gravado sobre o argumento recebido, e `fetch` permanecia marcado como concluído. O pipeline seguia com os pesos do modelo anterior sem qualquer aviso.~~ ✅ divergência aborta a execução; `--restart` descarta estado e artefatos de forma explícita
- ~~Os logits pré-computados eram reaproveitados na retomada apenas pela existência do diretório. Alterar `--top-k`, `--seq-len` ou o lote entre execuções reutilizava logits incompatíveis com a nova configuração.~~ ✅ cada etapa grava a impressão digital dos parâmetros que a geraram e é refeita quando eles mudam
- ~~`--target-params` não era validado contra `Budget.max_params_for_training()`. Um alvo acima do teto da máquina era aceito, e a falta de memória só aparecia na etapa `recover`, depois do download e da pré-computação dos logits.~~ ✅ recusado no plano, com `--allow-oversized` para assumir o risco deliberadamente
- ~~O exemplo de dimensionamento deste README excedia o teto de treino declarado.~~ ✅ substituído por ensaio medido e limites FP32 explícitos
- ~~A verificação de espaço em disco nunca disparava: `run_all()` era sempre chamado sem `required_disk_bytes`. `stage_logits` estimava os GB necessários e informava, mas não confrontava o valor com o espaço livre nem abortava.~~ ✅ verificação antes de baixar e antes de pré-computar, com piso mínimo no `doctor`
- ~~`stack do pipeline` era classificado como aviso, não como falha. `aguardente run` prosseguia sem `torch` instalado e terminava em `ModuleNotFoundError` cru, já dentro da execução.~~ ✅ aviso no `doctor`, bloqueio no `run`
- ~~`Machine.detect()` media sempre o volume `/`. Com `-o` apontando para disco externo, o espaço reportado não correspondia ao destino real.~~ ✅ mede o volume do destino, mesmo que o diretório ainda não exista
- ~~`SYSTEM_HEADROOM_BYTES` era fixo em 6 GB. Em um Mac de 8 GB isso deixava 2 GB de orçamento, valor irreal para qualquer etapa.~~ ✅ margem proporcional (25% da RAM, entre 4 GB e 12 GB), preservando os 6 GB da máquina de referência de 24 GB
- ~~O dtype real do treino divergia do orçamento de memória.~~ ✅ recuperação e teto agora usam FP32 em todos os dispositivos
- ~~Não havia exclusão mútua sobre o diretório de execução: duas instâncias apontadas para o mesmo `-o` sobrescreviam o estado uma da outra.~~ ✅ lock consultivo com recuperação de lock órfão
- ~~Exceções fora de `AguardenteError` — incluindo falta de memória, disco cheio e dependência ausente — chegavam ao usuário como traceback.~~ ✅ mensagens próprias por família; traceback sob `AGUARDENTE_DEBUG=1`
- ~~`arch._present` considerava presente qualquer valor diferente de `None`. Um config denso que declarasse `num_experts: 0` seria recusado como MoE.~~ ✅
- ~~`qk_norm` era inferido apenas de `model_type` iniciado em `qwen3`, subestimando a contagem de outras arquiteturas com RMSNorm por cabeça.~~ ✅ chave explícita quando existe, heurística apenas como fallback
- ~~`tie_word_embeddings` assumia `False` quando a chave estava ausente do config, divergindo do padrão do `transformers` e errando a contagem pelo tamanho do tensor de embeddings.~~ ✅
- ~~`VISION_KEYS` incluía a chave genérica `visual`, com risco de falso positivo em um config denso que a usasse com outro significado.~~ ✅ só dispara quando o valor é uma subconfiguração
- ~~Pesos pré-quantizados (GPTQ, AWQ, FP8) passavam pelas guardas com todas as dimensões densas no topo e falhavam tarde, na poda ou na recuperação.~~ ✅ recusados junto de MoE e multimodais
- ~~`head_dim` era truncado em silêncio quando `hidden_size` não era múltiplo do número de cabeças e o config não o declarava.~~ ✅
- ~~A variável `DEVELOPER_DIR` não era considerada no diagnóstico, que podia sugerir uma correção desnecessária.~~ ✅ tem precedência, e é oferecida como alternativa sem `sudo` em máquinas gerenciadas
- ~~Um diretório ativo inexistente — comum após atualizar ou mover o Xcode — caía no ramo genérico do diagnóstico.~~ ✅ causa própria, com instrução direta
- ~~O timeout de compilação era fixo em uma hora, sem relação com o tamanho do modelo.~~ ✅ escala com o artefato, com piso de 30 minutos
- ~~O módulo `ui` fixava `sys.stdout` nos parâmetros padrão no momento do import, o que impedia a captura da saída nos testes.~~ ✅ destino resolvido na chamada
- ~~A suíte pulava os testes de `distill`, `surgery` e parte de `security` sem `torch`, sem sinalizar a lacuna no resultado.~~ ✅ aviso no cabeçalho e no resumo da execução
- ~~`aguardente doctor` reportava `Xcode: ausente` para qualquer falha, incluindo Xcode instalado porém não selecionado, licença pendente e timeout.~~ ✅
- ~~A instrução exibida na falha do Xcode era `xcode-select --install`, que instala os Command Line Tools — exatamente a causa mais comum do problema.~~ ✅
- ~~A falha de `coreai-build` era sempre atribuída à ausência do Metal Toolchain, mesmo quando a causa real era o diretório de desenvolvimento ativo.~~ ✅
- ~~O `stderr` dos comandos era descartado, sem qualquer forma de inspecionar o motivo real da falha.~~ ✅ `aguardente doctor --verbose`
- ~~O diagnóstico dependia de mensagens de erro em inglês e falharia sob locale traduzido.~~ ✅ `LC_ALL=C` nos subprocessos e detecção de licença pelo código de saída 69
- ~~O timeout de 10 s em `xcodebuild -version` era insuficiente na primeira execução após a instalação do Xcode.~~ ✅ elevado para 30 s
- ~~Configs sem `intermediate_size` produziam `KeyError` cru, porque apenas duas das cinco dimensões obrigatórias eram lidas dentro do bloco protegido.~~ ✅
- ~~Configs MoE e multimodais eram aceitos e contados pela fórmula densa, com erro de até −95% na contagem de parâmetros.~~ ✅ recusa explícita com a razão
- ~~`compile_aot` não tratava ausência de `xcrun` nem timeout, e `inspect_asset` reportava o `stderr` sem instrução corretiva.~~ ✅
- ~~Falhas de `coreai-build` não indicavam quando a causa era o toolchain ativo em vez do componente ausente.~~ ✅

---

## Licença

Software proprietário. Uso permitido para fins pessoais e não comerciais. Consulte o arquivo [`LICENSE`](LICENSE) para mais detalhes.

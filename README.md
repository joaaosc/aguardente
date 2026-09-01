<p align="center">
  <img src="docs/media/icon.svg" width="128" alt="aguardente">
</p>

<h1 align="center">aguardente</h1>

<p align="center">
  Poda estruturada e destilação de modelos de linguagem para Apple Silicon via Core AI.
</p>

---

`aguardente` reduz o tamanho de modelos de linguagem (LLMs) por **poda estruturada** (pruning) e recupera qualidade via **destilação**, gerando pacotes `.aimodel` prontos para execução no framework Core AI da Apple.

---

## Instalação

Via script de instalação:

```bash
curl -fsSL https://raw.githubusercontent.com/joaaosc/aguardente/main/install.sh | bash
```

Ou a partir de um clone do repositório:

```bash
git clone https://github.com/joaaosc/aguardente && cd aguardente && ./install.sh
```

> **Nota:** As dependências do Core AI exigem Python `>=3.11,<3.14`. O instalador configura o ambiente com a versão adequada automaticamente via `uv`.

---

## Uso

Execução completa do pipeline:

```bash
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.0e9 --effort high --measure
```

O comando executa todas as etapas e retoma do ponto onde parou caso seja interrompido.

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
| **fetch** | Download dos arquivos necessários via `aria2c` com suporte a retomada. |
| **prune** | Avaliação de importância e poda estruturada de MLP, cabeças de atenção e camadas. |
| **logits** | Pré-computação dos top-k logits do modelo original (teacher) para o conjunto de calibração. |
| **recover** | Treinamento de destilação para recuperação de qualidade do modelo podado. |
| **export** | Conversão para o formato `.aimodel` compatível com Core AI. |

### Estratégia de poda

A redução segue a proporção típica de parâmetros na arquitetura do modelo (em que camadas MLP concentram a maior parte dos parâmetros):

1. **`intermediate_size` (MLP):** Redução dos neurônios intermediários, alinhada em múltiplos de 128 para compatibilidade com quantização per-block.
2. **Cabeças de atenção:** Redução preservando grupos completos de Grouped-Query Attention (GQA).
3. **Camadas:** Remoção de camadas intermediárias baseada em *block influence*, preservando sempre a primeira e a última camada.

A dimensão `hidden_size` é mantida inalterada para preservar a consistência de embeddings, projeções e normalizações.

### Esforço da destilação

`--effort` define **quanto trabalho** se investe para recuperar a qualidade perdida na poda. É um eixo ortogonal ao alvo: `--target-params` decide o tamanho do resultado, `--effort` decide o cuidado com que se chega nele. Um alvo agressivo com esforço baixo é o caminho mais curto para um modelo pequeno e ruim.

| Nível | | Épocas | top-k | Sequência | Logits em disco | Custo |
|---|---|---:|---:|---:|---:|---:|
| `low` | ▁··· rápido | 1 | 64 | 256 | 12 MB | 0,07× |
| `medium` | ▁▃·· equilibrado | 2 | 128 | 512 | 192 MB | 1,00× |
| `high` | ▁▃▅· cuidadoso | 3 | 192 | 768 | 864 MB | 4,26× |
| `max` | ▁▃▅▇ exaustivo | 4 | 256 | 1.024 | 3,0 GB | 14,74× |

O custo é o tempo relativo ao nível `medium`, calculado pelos tokens processados em cada etapa; o disco assume `--batch-size 2`. O nível `medium` é o padrão e reproduz exatamente a configuração usada antes de a escala existir.

Cada nível ajusta lotes de calibração, comprimento de sequência, lotes e profundidade dos logits do teacher, épocas, taxa de aprendizado, peso da destilação e acumulação de gradiente. `--batch-size` fica de fora de propósito: é restrição de memória da máquina, não escolha de qualidade.

Uma opção informada na linha de comando sempre vence o preset — `--effort max --epochs 1` usa tudo do nível exaustivo, com uma época só, e o painel diz o que foi sobrescrito.

```bash
aguardente effort          # compara os níveis, sem tocar em nenhum modelo
```

---

## Exemplo de dimensionamento

Exemplo de redução do Qwen3-4B para alvo de 1,0 B parâmetros, dimensionado para caber no treino de uma máquina de 24 GB:

|  | Original | Podado | Recuperado + int4 |
|---|---|---|---|
| Parâmetros | 4,02 B | 0,99 B | 0,99 B |
| Tamanho em disco | 7,49 GB | 1,85 GB | **0,53 GB** |
| Redução | — | 4,1× | **14,2×** |

O alvo não é livre: a etapa de recuperação precisa manter pesos, gradientes e os dois momentos do AdamW em memória, o que dá 12 bytes por parâmetro. Em 24 GB de RAM o orçamento é de 18 GB e o teto de treino fica em 1,21 B parâmetros — por isso o exemplo usa 1,0 B, e não um valor maior. `aguardente plan` calcula esse teto para a máquina local, e `aguardente run` recusa alvos acima dele.

A flag `--measure` avalia a perplexidade no conjunto de teste antes da poda, após a poda e após a recuperação.

---

## Requisitos

| Item | Requisito |
|---|---|
| Sistema | macOS 27+ com Apple Silicon |
| Xcode | Xcode 27+ com Metal Toolchain |
| Memória | 24 GB de RAM para modelos de ~4 B; 16 GB para modelos menores |
| Armazenamento | ~30 GB de espaço livre |

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

- O orçamento de treino cobre pesos, gradientes e estados do otimizador (12 bytes por parâmetro), sem termo para as ativações, que dependem de lote, comprimento de sequência e profundidade. A fração `train_fraction = 0,75` funciona como margem implícita, mas não foi calibrada contra medição: a folga real entre o teto e um alvo próximo dele pode ser menor que a aparente.
- O macOS não falha imediatamente sob pressão de memória, ele pagina. Uma execução acima do orçamento tende a ficar ordens de grandeza mais lenta em vez de abortar, e o sintoma é difícil de atribuir ao alvo escolhido.
- Um download interrompido de `xcodebuild -downloadComponent MetalToolchain` pode deixar `xcrun --find coreai-build` bem-sucedido com o componente incompleto. O diagnóstico aprovaria o ambiente e a falha só apareceria na compilação.
- A verificação de versão do Xcode aceita qualquer `27.x`, incluindo betas em que o Metal Toolchain ainda não está disponível. Na prática a verificação de `coreai-build` cobre o caso, mas o diagnóstico de versão sozinho não distingue.
- Os ramos de diagnóstico do Xcode foram validados por simulação, não em uma máquina realmente sem Xcode, com licença pendente ou com *first launch* incompleto. As mensagens de `stderr` usadas nos testes vêm de conhecimento prévio, não de captura em campo — `aguardente doctor --verbose` existe para coletar as reais.
- Não foram auditados: `fetch.py` além de `require_aria2` (retomada, verificação de integridade, modelos de acesso restrito), `probe.py`, `calibration.py`, `verify.py`, `distill/train.py`, `distill/loss.py` e a poda estruturada em `prune`.

### Corrigidos

- ~~Reutilizar um diretório de execução com outro modelo ou outro alvo era silenciosamente ignorado: `RunState.load_or_create` dava precedência ao valor gravado sobre o argumento recebido, e `fetch` permanecia marcado como concluído. O pipeline seguia com os pesos do modelo anterior sem qualquer aviso.~~ ✅ divergência aborta a execução; `--restart` descarta estado e artefatos de forma explícita
- ~~Os logits pré-computados eram reaproveitados na retomada apenas pela existência do diretório. Alterar `--top-k`, `--seq-len` ou o lote entre execuções reutilizava logits incompatíveis com a nova configuração.~~ ✅ cada etapa grava a impressão digital dos parâmetros que a geraram e é refeita quando eles mudam
- ~~`--target-params` não era validado contra `Budget.max_params_for_training()`. Um alvo acima do teto da máquina era aceito, e a falta de memória só aparecia na etapa `recover`, depois do download e da pré-computação dos logits.~~ ✅ recusado no plano, com `--allow-oversized` para assumir o risco deliberadamente
- ~~O exemplo de dimensionamento deste README (Qwen3-4B para 1,4 B) excedia o teto de treino da configuração declarada em *Requisitos*.~~ ✅ refeito com alvo de 1,0 B e o cálculo do teto explicado
- ~~A verificação de espaço em disco nunca disparava: `run_all()` era sempre chamado sem `required_disk_bytes`. `stage_logits` estimava os GB necessários e informava, mas não confrontava o valor com o espaço livre nem abortava.~~ ✅ verificação antes de baixar e antes de pré-computar, com piso mínimo no `doctor`
- ~~`stack do pipeline` era classificado como aviso, não como falha. `aguardente run` prosseguia sem `torch` instalado e terminava em `ModuleNotFoundError` cru, já dentro da execução.~~ ✅ aviso no `doctor`, bloqueio no `run`
- ~~`Machine.detect()` media sempre o volume `/`. Com `-o` apontando para disco externo, o espaço reportado não correspondia ao destino real.~~ ✅ mede o volume do destino, mesmo que o diretório ainda não exista
- ~~`SYSTEM_HEADROOM_BYTES` era fixo em 6 GB. Em um Mac de 8 GB isso deixava 2 GB de orçamento, valor irreal para qualquer etapa.~~ ✅ margem proporcional (25% da RAM, entre 4 GB e 12 GB), preservando os 6 GB da máquina de referência de 24 GB
- ~~Sem MPS disponível, `loading.pick_dtype` seleciona `float32`, dobrando o consumo de memória em relação ao orçamento, que assume 2 bytes por parâmetro.~~ ✅ o teto de treino acompanha o dtype do dispositivo e a queda é anunciada
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

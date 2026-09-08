# Auditoria do pipeline — 5 de setembro de 2026

Este documento preserva o estado e as medições daquela auditoria. As correções e os experimentos posteriores estão em [Melhorias da recuperação — 7 de setembro](recovery-improvements.md); algumas limitações de implementação descritas abaixo foram corrigidas depois.

**O pipeline agora executa treinamento real e produziu um `.aimodel` validado no Core AI do M1. O resultado de linguagem deste ensaio não foi bom.** A recuperação melhorou pouco o modelo podado; a conversão em 8 bits preservou esse resultado, incluindo suas repetições. Não há evidência para prometer que escolher parâmetros e aumentar `effort` torna cortes arbitrários úteis, nem que modelos pesados possam ser destilados inteiramente em 8 GB.

## Escopo e evidências

Foram examinados o código, o histórico das correções e a [sessão registrada](../.local/outputs.md). O ensaio usou [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), um modelo causal base, para permitir recuperação FP32 no M1 de 8 GB. Não foi usado o Qwen3-1.7B sugerido; não foi testada uma variante instruct, nem uma extração multimodal real nesta auditoria.

Ambiente: macOS 27, Xcode 27 beta/Metal Toolchain, Python 3.12, PyTorch 2.9.0, Transformers 5.12.1, coreai-core 1.0.0b2, coreai-torch 0.4.2, coreai-models 0.1.0 e datasets 4.8.5. Versões e hashes dos pesos estão em [provenance.json](../.local/audit-smollm/provenance.json). As evidências grandes permanecem em `.local`, fora do Git.

Verificação final: **570 testes passaram**, incluindo regressões de gradiente fora do top-k, atualização real de pesos, acumulação residual, padding, metadados por camada, rejeição de modelo não causal e reprovação de export quando o runtime falha. [Log dos testes](../.local/audit-smollm/tests-final.log). `uv build --wheel` concluiu; a distribuição contém a ponte Swift e os módulos de exportação/validação, conferidos byte a byte contra o fonte. `git diff --check` passou. Esses testes verificam comportamento do código, não qualidade linguística.

## As correções anteriores eram legítimas?

| Alteração anterior | Veredito | Tratamento nesta revisão |
|---|---|---|
| `afe3bbe`: campo `discard_source_weights` ausente | Correção real de contrato entre CLI e pipeline. | Mantida. |
| `b27ba8f`: criação tardia do estado | Correção real de efeito colateral; mudar um teste para o contrato correto não é, por si, bypass. | Mantida. |
| `00d3080`: encurtar listas do config após remover camadas | Defeito real, corrigido com heurística perigosa: qualquer lista do mesmo comprimento podia ser alterada. | Agora somente campos conhecidos de configuração por camada são recortados. Teste preserva uma lista de tokens EOS. |
| `b236d42`: trocar FP16 por BF16 e conferir finitude | Evita overflow, mas finitude não prova aprendizado: pesos e momentos BF16 perdem atualizações pequenas. | Recuperação promove pesos, gradientes e AdamW a FP32; orçamento passa a 16 bytes estáticos por parâmetro. |
| `898f72f`: fabricar cache offline do Hub para exportar diretório local | Contorno da interface de download, sem evidência de adulteração dos pesos; não corrigia suporte local do exportador. | Adaptador usa APIs locais da implementação Apple; não fabrica identidade/snapshot do Hub. |
| `a3c3cee`: silenciar saída das importações | Supressão de diagnóstico, não correção das incompatibilidades avisadas. | Saída capturada volta a ser emitida em stderr. |
| `34ae920`: ajustar mensagem de extração | Correção de comunicação. | Mantida; não demonstra correção numérica. |

O modelo de embeddings usado na sessão anterior não era uma base válida para esse teste causal. Instanciar uma cabeça de linguagem ausente não cria uma cabeça treinada. Agora arquiteturas declaradamente de embeddings/classificação são recusadas, e o carregamento causal rejeita pesos ausentes ou incompatíveis. A guarda ainda não prova a semântica de todo config desconhecido.

## Experimento reproduzível

O download real foi executado pela CLI. A execução seguinte consumiu esse diretório local; a indicação de `fetch` como dispensado no estado final significa que os arquivos já estavam locais. Nenhuma etapa necessária de poda, logits, recuperação ou conversão foi desativada. O primeiro export usou int4; o último comando retomou o mesmo treinamento concluído e reexportou em int8.

Para repetir com o código deste checkout e um Python contendo a extra `pipeline`:

```bash
PIPELINE_PY=/Users/joaocosta/.local/share/uv/tools/aguardente/bin/python
"$PIPELINE_PY" scripts/dev_cli.py fetch HuggingFaceTB/SmolLM2-135M \
  -o .local/audit-smollm/teacher
"$PIPELINE_PY" scripts/dev_cli.py run .local/audit-smollm/teacher \
  -o .local/audit-smollm/run --target-params 120e6 --effort medium \
  --batch-size 1 --seq-len 256 --logit-batches 32 --calib-batches 8 \
  --epochs 2 --grad-accum 4 --max-context-length 512 --measure --device mps \
  --compression-config docs/recipes/smollm2-int8.yaml
```

Na execução registrada, a receita tinha o caminho `.local/audit-smollm/int8.yaml`; a cópia versionada acima contém a mesma configuração. Para nova execução use outro diretório de saída. O comando reproduz o procedimento, não garante pesos idênticos: revisão do Hub, semente e estados RNG ainda não são fixados integralmente pelo pipeline.

Foram usados 16 exemplos para pontuar a poda e 64 para produzir logits, com **9.435 transições de tokens não mascaradas distintas**, duas épocas e 32 passos do AdamW. A recuperação levou 233,62 segundos. São overrides do preset `medium`, não uma avaliação do preset completo. `intermediate_size` caiu de 1536 para 1152; 30 camadas, largura 576, 9 cabeças de query e 3 grupos KV foram preservados. O alvo é um teto sujeito a alinhamento, portanto 120 milhões resultaram em 114.608.448 parâmetros.

### Qualidade da poda e recuperação

Protocolo: primeiros 20.000 caracteres do split test do WikiText-2, janela 512, stride 256. Teacher inferido em BF16; recuperação em FP32. Menor perplexidade é melhor.

| Modelo | Parâmetros | Perplexidade |
|---|---:|---:|
| Original | 134.515.008 | 17,954 |
| Podado | 114.608.448 | 32,113 |
| Recuperado | 114.608.448 | 30,304 |

A redução foi de **14,8% dos parâmetros**, mas a perplexidade final ficou **68,8% acima da original**. A recuperação devolveu somente **12,8% da queda**, usando a fração `(PPL_podado − PPL_recuperado) / (PPL_podado − PPL_original)`. Isso não é percentual de capacidade recuperada. Evidências: [execução](../.local/audit-smollm/run.log), [estado](../.local/audit-smollm/run/state.json) e [recuperação](../.local/audit-smollm/run/ckpt/recovery.json).

### Fidelidade da conversão

Protocolo separado: primeiros 6.000 caracteres do mesmo split test, 1.549 tokens avaliados, janela 256, stride 128. Todas as referências PyTorch desta comparação são FP32. Não comparar diretamente esta tabela à anterior.

| Variante | Tamanho do `.aimodel` | Perplexidade |
|---|---:|---:|
| Teacher original, PyTorch | — | 19,752 |
| Student recuperado, PyTorch | — | 32,029 |
| Student Core AI FP16, sem quantização | 218,90 MiB | 32,019 |
| Student Core AI int4 padrão | 61,87 MiB | 43,964 |
| Student Core AI int8 | **116,50 MiB** | **32,039** |

Int4 piorou a perplexidade em 37,3% contra o student PyTorch e **foi reprovado**, com código de saída 1. Não foram relaxadas tolerâncias para aprová-lo. Int8 variou cerca de 0,033% nesse recorte. Nos três prompts, int8 apresentou erro RMS relativo de 3,54–4,07%, similaridade cosseno acima de 0,99926 e concordância de argmax de 86,8–95,8%. FP16 teve erro RMS relativo de 0,2–0,5% e concordância de argmax de 100%.

Também foram executados prefill, decode incremental, comparação do KV cache com forward completo e geração greedy de até 32 tokens. As três continuações int8 coincidiram exatamente com PyTorch, inclusive a repetição de “You learn English just as you learn a language” e “que que que”. O teacher original produziu continuações melhores nos dois prompts em inglês; também era fraco no prompt em português. Três exemplos não constituem benchmark de idioma ou de conversa.

Evidências: [validação final int8](../.local/audit-smollm/run/runtime-validation.json), [CLI final](../.local/audit-smollm/run-int8.log), [int4 reprovado](../.local/audit-smollm/runtime-swift-4bit-full.json), [FP16](../.local/audit-smollm/runtime-swift-fp16.json), [teacher](../.local/audit-smollm/teacher-baseline.json).

### Falha nativa e contrato de runtime

O `.aimodel` inicial exportava, mas o primeiro forward pela API Python causava SIGSEGV no compilador ANE, com `ANE regionCall op not found`. Preferir GPU sozinho não resolveu; FP16 também falhou. O SDK Swift instalado oferece `SpecializationOptions.expectFrequentReshapes`, ausente no wrapper Python instalado. Com essa opção verdadeira e preferência GPU, o mesmo asset executou pelo framework CoreAI do sistema.

O projeto agora contém uma ponte Swift que expressa esse contrato para sequências dinâmicas. Ela não substitui o modelo, não ignora operadores e não captura uma falha para devolver uma resposta falsa. **Não foi corrigido o compilador da Apple**: foi corrigida a integração com a API disponível, validada neste M1/SDK. O caminho padrão Python continua com a falha registrada. `--measure` agora executa a validação real após exportar e reprova a etapa se o processo retornar erro.

Houve ainda compilação AOT bem-sucedida para 20 arquiteturas, registrada em [compile.log](../.local/audit-smollm/compile.log); a execução validada carregou o `.aimodel` diretamente. Não foi demonstrada execução do `.aimodelc` standalone, nem medido ganho de latência, pico de RAM ou uso exclusivo de ANE.

## Revisão etapa por etapa

1. **Download e identificação.** O caminho causal local funciona e deixou evidência verificável. Falta conferir hashes LFS automaticamente e fixar revisão do Hub. Tokenizer de student externo é validado apenas por tamanho do vocabulário: índices diferentes com igual tamanho invalidam KD. Exigir igualdade do mapeamento e dos tokens especiais é uma correção necessária antes de confiar nesse modo.

2. **Planejamento.** Contagem e alinhamentos determinam um modelo estruturalmente possível; não predizem qualidade. A política corta MLP antes de atenção/profundidade, sem comparar arquiteturas de mesmo orçamento. O teto de treino usa FP32 e uma estimativa mais conservadora de ativações, mas continua sem calibração de pico real. O teacher inteiro precisa caber antes da poda. Rodar o resultado num Air e produzir esse resultado no Air são requisitos diferentes; modelos grandes exigirão preparação numa máquina maior ou um novo regime de treino.

3. **Dados e pontuação.** Corrigidos scores que compartilhavam índices entre camadas, influência do padding e ponderação dos neurônios MLP pela norma de seus pesos de saída. A pontuação agora é por camada/token ativo. A importância de atenção ainda é uma heurística da ativação de V; não mede a perda causal ao remover cada grupo. O corpus sequencial e pequeno pode deixar o corte frágil fora do domínio. O próximo passo é dados representativos, embaralhamento, packing e ablações de cortes, antes de aumentar agressividade.

4. **Cirurgia.** O ensaio confirmou redução real de pesos e carregamento correto; não apenas zeros ou metadados de tamanho. Foram corrigidas listas do config. Este ensaio cortou somente MLP: testes sintéticos de cabeças e camadas não equivalem a validação de qualidade desses cortes em modelos reais. Extração multimodal também precisa de paridade com o decoder original.

5. **Logits do teacher.** Antes, normalizar apenas o top-k do student não penalizava sua probabilidade fora desse conjunto. Agora shards guardam logits FP32, índices int32, temperatura e logsumexp sobre todo o vocabulário. A loss compara `KL(teacher || student)` em K categorias mais uma categoria agregada para a cauda, com normalização completa do student. É uma aproximação coerente, **não KL exata no vocabulário inteiro**: a estrutura interna da cauda se perde. Shards antigos sem normalizador são recusados na recuperação.

6. **Otimização.** Há forward, backward e mudanças reais dos pesos. Corrigidos FP32 do AdamW, primeiro passo com LR não nula, escala do lote residual e recusa de gradientes não finitos. Seleção usa validation; test fica para a avaliação final. Ainda há duas limitações: acumulação pondera médias de lotes, não o total de tokens válidos; e o limiar de 0,5% de melhora para early stopping também controla salvar o melhor checkpoint. Neste ensaio salvou/restaurou o passo 24, embora o passo 32 tivesse PPL de validação ligeiramente menor (23,316 contra 23,403). Separar salvar o mínimo real de decidir paciência é a correção adequada. Poucas épocas sobre os mesmos 9.435 tokens não substituem dados novos.

7. **Exportação e quantização.** O adaptador local usa o carregador de modelos Apple, quantização e exportação Apple; para Llama cria uma configuração local canônica compatível com o adaptador Mistral. A equivalência foi conferida numericamente neste SmolLM2. Não há prova universal para arquiteturas adicionais. O caminho local atual é macOS; iOS é recusado explicitamente. O carregamento direto tem pico de RAM maior que streaming. Int4 deve ser tratado como candidato a medir, não qualidade garantida; int8 resolveu a degradação adicional neste caso. A receita preserva um override int4 de SwitchLinear do padrão Apple, irrelevante para este modelo denso.

8. **Runtime e aceitação.** O teste passou no framework nativo, com caches persistentes, e detectou a quantização ruim. As tolerâncias numéricas são critérios de engenharia deste projeto, não padrões publicados da Apple nem selo de qualidade linguística. O validador cobre contextos curtos e FP16/FP32; a ponte não implementa cache BF16, não tem timeout por chamada e não substitui testes de contexto longo/concorrência. Perplexidade e geração são registradas, mas a aprovação automática ainda se baseia na paridade dos logits/cache dos prompts, não numa exigência de qualidade contra o teacher.

9. **Retomada e produto.** Fingerprints e checkpoints evitam vários reaproveitamentos incorretos; logs agora preservam avisos. Ainda faltam fingerprints do conteúdo dos dados/receitas/pesos e RNG completo. Reexportar configurações diferentes que colidam no mesmo nome de asset exige novo destino; o exportador local recusa sobrescrita. A escolha do bundle por metadata mais recente também deve ser substituída por identidade explícita do artefato. São limitações concretas, não justificativa para declarar um resultado degradado como bom.

## O processo é adequado e útil?

Poda estruturada seguida de KD é uma estratégia legítima, empregada em trabalhos como [Minitron](https://arxiv.org/abs/2407.14679). Isso não valida automaticamente esta implementação, seus dados ou seus presets. Projetos como [Sheared LLaMA](https://xiamengzhou.github.io/sheared-llama/) empregam treinamento continuado e seleção de dados em escala muito maior; não se deve transformar esse contraste numa regra universal de quantos tokens bastam.

O `medium` completo permite no máximo 262.144 posições únicas antes de padding; `max`, cerca de 2,1 milhões. Esses nomes expressam orçamento de execução, não qualidade comprovada. Neste ensaio reduzido, a perda observada foi grande para um corte modesto. A ferramenta é uma **base experimental funcional**, ainda sem evidência de destilação automática suficientemente boa para sua promessa principal.

Antes de declarar uma versão pronta, eu exigiria comparação contra o modelo original apenas quantizado e contra um modelo menor nativo, em tarefas e idiomas de uso real, com limite explícito de degradação, memória e latência medidos no destino. Depois: dados/recuperação suficientes para atingir esse limite, seleção de checkpoint correta, tokenizers comprovadamente compatíveis e quantização escolhida pelo resultado medido. Esses trabalhos são propostas após esta auditoria; não foram apresentados como concluídos.

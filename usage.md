# Guia de uso

Documento de referência: o que cada comando faz, o que esperar de cada etapa,
e o que fazer quando algo falha.

- [Antes de começar](#antes-de-começar)
- [Ensaio recomendado](#ensaio-recomendado)
- [Os comandos](#os-comandos)
- [Anatomia de uma execução](#anatomia-de-uma-execução)
- [Retomada](#retomada)
- [Quando a memória não dá](#quando-a-memória-não-dá)
- [Tratamento de erros](#tratamento-de-erros)
- [Perguntas frequentes](#perguntas-frequentes)

---

## Antes de começar

```bash
aguardente doctor
```

Nove verificações. Todas precisam passar antes de qualquer trabalho pesado —
descobrir um requisito ausente depois de quarenta minutos de download é o
desperdício mais comum.

<!-- placeholder: captura da saída de `aguardente doctor` -->
![Verificação de ambiente](docs/media/doctor.png)

| Verificação | Por que importa |
|---|---|
| macOS 27+ | Requisito do framework Core AI |
| Xcode 27+ | Traz o compilador de modelos |
| `coreai-build` | Vem com o Metal Toolchain |
| `aria2c` | Downloads retomáveis; sem ele, uma queda de rede custa o download inteiro |
| Python 3.11–3.13 | O stack Core AI recusa 3.14 |
| Apple Silicon | Não há binários para Intel |
| Stack do pipeline | `torch`, `transformers` e os pacotes Core AI |
| RAM e disco | Orçamento disponível |

Cada linha que falha traz a correção junto.

---

## Ensaio recomendado

**Faça isto antes do modelo grande.** Percorre o caminho inteiro com um modelo
pequeno em poucos minutos. Se algo estiver quebrado, você descobre agora.

```bash
aguardente run HuggingFaceTB/SmolLM2-135M-Instruct \
    -o ensaio --target-params 90e6 \
    --calib-batches 4 --logit-batches 8 --epochs 1 \
    --export-dry-run
```

Baixa 256 MB, poda, recupera e **valida a configuração da conversão sem
converter**. O `--export-dry-run` é o teste mais barato que existe para o
único risco real do pipeline.

Passando, repita sem a flag para produzir um `.aimodel` de verdade.

---

## Os comandos

### `plan` — o que vai acontecer

```bash
aguardente plan Qwen/Qwen3-4B [--target-params 1.4e9]
```

Não baixa um único byte. Consulta os metadados do repositório e o
`config.json`, e a partir daí calcula tudo: distribuição de parâmetros, plano
de poda, tamanho estimado do resultado, KV cache e RAM de treino.

Sem `--target-params`, o alvo é derivado da RAM da máquina.

### `fetch` — só o download

```bash
aguardente fetch Qwen/Qwen3-4B -o teacher/ [--connections 8]
```

Usa `aria2` com várias conexões. Baixa apenas o que o pipeline consome —
ignora READMEs, licenças e pesos em formatos alternativos. Num repositório com
`.bin` **e** `.safetensors`, isso corta metade do tráfego.

Idempotente: arquivos completos são pulados, truncados são retomados do ponto
onde pararam.

### `run` — tudo

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
| Conversão | `--platform macOS` `--compression 4bit` `--max-context-length` `--export-dry-run` |
| Controle | `--device` `--measure` `--skip-recover` `--skip-export` `--skip-checks` |

`--measure` mede a perplexidade em três pontos e é o que dá lastro à
afirmação "recuperou X% da queda". Custa tempo; vale.

### `status` — onde parou

```bash
aguardente status -o run/qwen3
```

---

## Anatomia de uma execução

### 1. fetch

Baixa o modelo original. O tempo depende inteiramente da rede.

### 2. prune

Carrega o modelo, roda alguns lotes de texto real para medir a importância de
cada unidade, corta, e **verifica que o resultado ainda produz números
finitos** antes de gravar. Um modelo cortado que não roda não vale a pena
gravar.

Três eixos são cortados em ordem de prioridade: neurônios do MLP, cabeças de
atenção (em grupos GQA inteiros) e camadas inteiras.

### 3. logits

Roda o modelo original sobre o conjunto de calibração e guarda apenas as 128
previsões mais prováveis por posição. Depois **descarrega o original da
memória**.

Por que só as 128 maiores: guardar as previsões completas para 10 mil amostras
de 512 tokens com vocabulário de 152 mil custaria cerca de 1,4 TB. Com o
recorte, cai para poucos GB — e a cauda descartada quase não influencia o
resultado.

### 4. recover

A etapa longa. O modelo podado é treinado a reproduzir as previsões do
original. Não está aprendendo do zero: está reencontrando um equilíbrio que a
poda desfez, e por isso os hiperparâmetros são de ajuste fino.

Para sozinha quando a melhora satura. `Ctrl-C` grava o progresso.

### 5. export

Delega ao `coreai.llm.export` da Apple. Produz um diretório com o `.aimodel`,
o tokenizer e os metadados.

---

## Retomada

Cada etapa grava o que produziu em `state.json`. Interromper e repetir o mesmo
comando pula tudo que já terminou.

```bash
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.4e9   # cai na etapa 4
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.4e9   # retoma da 4
```

Uma etapa marcada como "em curso" num estado lido do disco é órfã — o processo
anterior morreu no meio — e volta para pendente automaticamente.

Estrutura da execução:

```
run/qwen3/
├── state.json      progresso por etapa
├── teacher/        modelo original
├── pruned/         modelo podado
├── logits/         previsões pré-computadas
├── student/        modelo recuperado
├── ckpt/           checkpoints do treino
└── bundle/         .aimodel final
```

---

## Quando a memória não dá

Em ordem, do mais barato ao mais custoso em qualidade:

| Ajuste | Efeito |
|---|---|
| `--grad-accum 8` | Mesmo lote efetivo, menos memória por passo |
| `--seq-len 256` | Ativações caem quase pela metade |
| `--batch-size 1` | O mínimo |
| `--target-params` menor | Modelo menor treina com menos memória |
| `--skip-recover` | Só poda, sem treino — resultado pior, mas cabe |

O `plan` avisa antecipadamente quando o estado de treino excede o orçamento.

---

## Tratamento de erros

Toda mensagem de erro traz a causa e a ação sugerida. Os casos mais comuns:

### Ambiente

| Mensagem | Causa | Solução |
|---|---|---|
| `Core AI exige macOS 27.0 ou superior` | Sistema antigo | Atualizar. Poda e destilação funcionam mesmo assim; só a conversão exige |
| `coreai-build não encontrado` | Metal Toolchain ausente | `xcodebuild -downloadComponent MetalToolchain` |
| `aria2c não encontrado` | Sem o acelerador de download | `brew install aria2` |
| `coreai-opt exige >=3.11,<3.14` | Python errado | `uv venv --python 3.12` — o instalador já fixa isso |
| `coreai-core só publica wheels para macOS arm64` | Mac Intel | Sem solução: o runtime não existe para Intel |

### Download

| Mensagem | Causa | Solução |
|---|---|---|
| `aria2c saiu com código N` | Rede caiu | Repetir o comando; retoma do ponto onde parou |
| `N arquivo(s) incompletos após o download` | Transferência truncada | Repetir; o tamanho errado é detectado e corrigido |
| `acesso negado` / `Modelo gated` | Licença não aceita | Aceitar na página do modelo e rodar `hf auth login` |
| `não publica pesos em .safetensors` | Formato antigo | Não suportado; escolher outro modelo |

### Poda e treino

| Mensagem | Causa | Solução |
|---|---|---|
| `o modelo podado produz NaN ou Inf` | Corte agressivo demais | Alvo maior com `--target-params` |
| `alvo é menor que o piso de poda` | Alvo abaixo do que a arquitetura permite | As embeddings sozinhas já podem exceder o alvo; escolher modelo menor |
| `num_attention_heads não é múltiplo de num_key_value_heads` | Arquitetura fora do padrão | Não suportada |
| `não foi possível localizar a lista de camadas` | Não é um modelo causal do formato transformers | Não suportado |
| `MPS backend out of memory` | Sem memória | `--grad-accum 8 --seq-len 256` |

### Calibração

| Mensagem | Causa | Solução |
|---|---|---|
| `não foi possível carregar dados de calibração` | Rede, ou identificador de dataset mudou | `--calib-file meu.txt` com parágrafos separados por linha em branco |
| `Repository id must be 'namespace/name'` | Identificador antigo sem namespace | Já corrigido internamente; se voltar, usar `--calib-dataset namespace/nome` |

### Conversão

| Mensagem | Causa | Solução |
|---|---|---|
| `coreai.llm.export não encontrado` | Stack Core AI ausente | Reexecutar `./install.sh` |
| `coreai.llm.export falhou (código N)` | Ferramenta da Apple recusou a entrada | Repetir com `--export-dry-run` para ver a validação isolada |

> Sobre `coreai-models`: o nome está ocupado no índice público do Python por um
> pacote de terceiro, sem autor declarado. O pacote real da Apple existe apenas
> no repositório oficial, e o instalador aponta para lá explicitamente. Não
> instale `coreai-models` pelo índice público.

---

## Perguntas frequentes

**Poda ou quantização — qual usar?**
As duas, nesta ordem. A quantização é gratuita: não precisa de dados nem
treino, e corta o tamanho por cerca de 4×. Só recorra à poda quando ela
sozinha não bastar, porque poda custa horas de treino.

**Por que o modelo podado fica pior antes de melhorar?**
Porque é isso que acontece. A poda remove capacidade e a perplexidade sobe; a
destilação recupera parte dela. Com `--measure` você vê os três números e
julga se o resultado serve.

**Posso interromper?**
Sim. `Ctrl-C` grava o estado. Repetir o comando retoma.

**Preciso de GPU NVIDIA?**
Não. O pipeline roda em Apple Silicon via MPS.

**Quanto tempo demora?**
Depende do modelo, da máquina e da rede. O download e a recuperação dominam.
Faça o ensaio primeiro para calibrar a expectativa.

**Onde fica o resultado?**
Em `<saída>/bundle/`. Menos de 1 GB para um alvo de 1,4 B parâmetros.

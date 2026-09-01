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
aguardente run Qwen/Qwen3-4B -o run/qwen3 --target-params 1.4e9 --measure
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

---

## Exemplo de dimensionamento

Exemplo de redução do Qwen3-4B para alvo de 1,4 B parâmetros:

|  | Original | Podado | Recuperado + int4 |
|---|---|---|---|
| Parâmetros | 4,02 B | 1,37 B | 1,37 B |
| Tamanho em disco | 7,49 GB | 2,55 GB | **0,73 GB** |
| Redução | — | 2,9× | **10,2×** |

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

## Licença

Software proprietário. Uso permitido para fins pessoais e não comerciais. Consulte o arquivo [`LICENSE`](LICENSE) para mais detalhes.

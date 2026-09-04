# Avisos de terceiros

O aguardente é distribuído sob licença de uso restrito (veja `LICENSE`). Isso é
compatível com as dependências abaixo: todas são permissivas — BSD 3-Clause ou
Apache 2.0 — e nenhuma delas é copyleft, de modo que nenhuma obriga a abrir o
código deste projeto. O que todas exigem é **atribuição**: o aviso de copyright
e o texto da licença precisam acompanhar qualquer redistribuição.

Este arquivo é essa atribuição, e o script `scripts/build-app.sh` o copia para
dentro de `Aguardente.app/Contents/Resources` justamente por isso.

## Dependências do pipeline

Instaladas pelo extra `pipeline` (veja `pyproject.toml`), não redistribuídas
junto do aplicativo — o `Aguardente.app` executa o CLI instalado à parte.

| Pacote | Licença | Titular |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | Meta Platforms, Inc. e colaboradores |
| [Transformers](https://github.com/huggingface/transformers) | Apache 2.0 | Hugging Face Inc. |
| [Datasets](https://github.com/huggingface/datasets) | Apache 2.0 | Hugging Face Inc. |
| [Accelerate](https://github.com/huggingface/accelerate) | Apache 2.0 | Hugging Face Inc. |
| [coreai-torch](https://github.com/apple/coreai) | BSD 3-Clause | Apple Inc. |
| [coreai-opt](https://github.com/apple/coreai) | BSD 3-Clause | Apple Inc. |
| [coreai-models](https://github.com/apple/coreai-models) | BSD 3-Clause | Apple Inc. |

## Ferramentas externas

Invocadas como processos, nunca vinculadas nem redistribuídas.

| Ferramenta | Licença | Observação |
|---|---|---|
| [aria2](https://aria2.github.io) | GPL 2.0+ com exceção OpenSSL | Executada como processo separado (`aria2c`), instalada pelo usuário via Homebrew. Chamar um programa GPL como subprocesso não torna este projeto uma obra derivada. |

## Modelos

O aguardente **não redistribui pesos de modelo**. Ele baixa o que o usuário
pedir e produz um modelo derivado a partir dele.

### A licença dos pesos não restringe a licença deste programa

Nenhuma licença de modelo alcança o código do aguardente. Elas governam os
pesos e as obras derivadas deles — não o software que os processa. Não existe
aqui o efeito de contaminação que uma licença copyleft teria sobre código: o
aguardente pode continuar sob licença de uso restrito independentemente de
quais modelos ele venha a converter.

O que a licença do modelo governa é **a saída do pipeline**. Poda e destilação
produzem obra derivada dos pesos de origem, e é a licença de origem que a
acompanha — não a deste programa. Em termos práticos, para quem for
redistribuir um modelo convertido:

| Modelo de origem | Licença dos pesos | Efeito sobre o derivado |
|---|---|---|
| Qwen 3 (todas as escalas) | Apache 2.0 | Permissiva: pode ser redistribuído sob os termos que você escolher, inclusive restritivos, preservando a atribuição. |
| Mistral 7B Instruct v0.3 | Apache 2.0 | Idem. |
| SmolLM2 | Apache 2.0 | Idem. |
| DeepSeek-R1-Distill-Qwen | MIT | Permissiva, com atribuição. |
| Llama 3.x | Llama Community License | Restrições de uso, obrigação de nomear o derivado começando com "Llama" e de propagar a licença. |
| Gemma | Gemma Terms of Use | Restrições de uso que se propagam a derivados. |

O modelo padrão do aplicativo — Qwen 3 — é Apache 2.0, o caso mais folgado:
nada nele limita a licença que você aplica ao resultado.

A regra geral fica: **a licença de uso restrito deste programa cobre o
programa, não os pesos que ele produz.** Ao distribuir um modelo convertido,
verifique a licença do modelo de origem.

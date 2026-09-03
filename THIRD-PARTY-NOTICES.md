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

Isso tem uma consequência que a licença deste projeto não resolve: **o modelo
que sai do pipeline herda os termos do modelo que entrou**. Poda e destilação
produzem uma obra derivada, e cada família tem regras próprias — Qwen e Mistral
publicam sob Apache 2.0, enquanto Llama e outras usam licenças de comunidade
com restrições de uso e obrigações de nomenclatura para derivados.

Quem for redistribuir um modelo gerado aqui precisa checar a licença do modelo
de origem. A licença de uso restrito deste programa cobre o programa, não os
pesos que ele produz.

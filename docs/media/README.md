# Mídia da documentação

Capturas referenciadas no `README.md` e no `usage.md`. Os arquivos não estão
no repositório ainda — este documento diz como gravá-los.

| Arquivo | Onde aparece | O que mostrar |
|---|---|---|
| `demo.gif` | README, topo | `aguardente run` do início ao fim, acelerado |
| `plan.png` | README, seção Uso | saída completa de `aguardente plan Qwen/Qwen3-4B` |
| `doctor.png` | usage.md | saída de `aguardente doctor` com tudo verde |

## Como gravar

**GIF animado** — `asciinema` grava o terminal como texto (arquivo leve, sem
perda de nitidez) e `agg` converte para GIF:

```bash
brew install asciinema agg
asciinema rec docs/media/demo.cast --cols 80 --rows 30
# ... execute o comando a ser demonstrado, depois Ctrl-D
agg docs/media/demo.cast docs/media/demo.gif --font-size 16 --speed 2
```

**Captura estática** — redirecionar para arquivo remove as cores. Para
preservá-las, use `script`:

```bash
script -q /dev/null aguardente plan Qwen/Qwen3-4B | tee /tmp/plan.txt
```

Depois capture a janela do terminal (`Cmd+Shift+4`, barra de espaço, clique).

## Recomendações

- Terminal com **80 colunas**; mais que isso quebra o enquadramento no GitHub.
- Tema escuro com bom contraste.
- Nada de caminhos pessoais, tokens ou nomes de usuário visíveis na captura.
- GIF abaixo de 5 MB — o GitHub não carrega arquivos maiores na visualização.

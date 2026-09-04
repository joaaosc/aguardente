"""Contenção do ruído de importação das dependências pesadas.

`torch`, `transformers` e os pacotes Core AI escrevem avisos direto no
descritor de erro no momento em que são importados — versão de torch não
testada, extensões C++ puladas, redirecionamentos indisponíveis no macOS.
São diagnósticos das bibliotecas, não da execução, e caem no meio dos painéis
formatados do CLI. A captura acontece no nível do descritor porque parte do
texto vem de código nativo, fora do alcance de `contextlib.redirect_stderr`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from collections.abc import Iterator


class Captured:
    """Texto contido durante o bloco, preenchido quando ele termina."""

    __slots__ = ("text",)

    def __init__(self) -> None:
        self.text = ""

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def __str__(self) -> str:
        return self.text


@contextlib.contextmanager
def silenced() -> Iterator[Captured]:
    """Desvia stdout e stderr para um buffer temporário durante o bloco.

    O objeto cedido só tem conteúdo depois que o bloco termina, e serve para
    anexar o ruído a um diagnóstico quando algo dentro dele falha. Exceções
    continuam propagando: só a escrita nos descritores é contida.
    """
    saida = Captured()
    try:
        fd_out, fd_err = sys.stdout.fileno(), sys.stderr.fileno()
        salvo_out, salvo_err = os.dup(fd_out), os.dup(fd_err)
    except (AttributeError, OSError, ValueError):
        # Sem descritores reais — stdout substituído por um buffer em memória,
        # como sob captura de teste. Nada nativo escreve ali de todo jeito.
        yield saida
        return

    with tempfile.TemporaryFile(mode="w+b") as buf:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(buf.fileno(), fd_out)
            os.dup2(buf.fileno(), fd_err)
            yield saida
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(salvo_out, fd_out)
            os.dup2(salvo_err, fd_err)
            os.close(salvo_out)
            os.close(salvo_err)
            buf.seek(0)
            saida.text = buf.read().decode("utf-8", "replace")

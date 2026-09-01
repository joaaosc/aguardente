"""Formatação e apresentação de interface no terminal."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from enum import Enum
from typing import Iterable, Sequence, TextIO

# --------------------------------------------------------------------- cor

def _color_enabled() -> bool:
    """Verifica se o ambiente suporta cores ANSI e se a saída é um TTY."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    return sys.stdout.isatty()


COLOR = _color_enabled()
WIDTH = min(shutil.get_terminal_size((80, 24)).columns, 78)


class _Style:
    """Estilos e cores ANSI para formatação no terminal."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    SUCCESS = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    ACCENT = "\033[36m"


def _paint(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    return f"{''.join(codes)}{text}{_Style.RESET}"


def bold(text: str) -> str:
    return _paint(text, _Style.BOLD)


def dim(text: str) -> str:
    return _paint(text, _Style.DIM)


def accent(text: str) -> str:
    return _paint(text, _Style.ACCENT)


def success(text: str) -> str:
    return _paint(text, _Style.SUCCESS)


def warning(text: str) -> str:
    return _paint(text, _Style.WARNING)


def failure(text: str) -> str:
    return _paint(text, _Style.ERROR)


def _visible_len(text: str) -> int:
    """Calcula o comprimento visível do texto ignorando códigos de escape ANSI."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
            continue
        out += 1
        i += 1
    return out


# ----------------------------------------------------------------- estados


class StageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


_MARKS: dict[StageState, tuple[str, str]] = {
    StageState.PENDING: ("·", ""),
    StageState.RUNNING: ("»", _Style.ACCENT),
    StageState.OK: ("✓", _Style.SUCCESS),
    StageState.FAILED: ("✗", _Style.ERROR),
    StageState.SKIPPED: ("–", _Style.DIM),
}

_LABELS: dict[StageState, str] = {
    StageState.PENDING: "pendente",
    StageState.RUNNING: "em curso",
    StageState.OK: "concluída",
    StageState.FAILED: "falhou",
    StageState.SKIPPED: "pulada",
}


# ------------------------------------------------------------- primitivas


def blank(out: TextIO = sys.stdout) -> None:
    print(file=out)


def rule(char: str = "─", out: TextIO = sys.stdout) -> None:
    """Imprime linha divisória horizontal."""
    print(dim(char * WIDTH), file=out)


def title(text: str, subtitle: str | None = None, out: TextIO = sys.stdout) -> None:
    """Exibe o cabeçalho principal do comando."""
    blank(out)
    print(bold(text), file=out)
    if subtitle:
        print(dim(subtitle), file=out)
    rule(out=out)


def header(text: str, out: TextIO = sys.stdout) -> None:
    """Exibe o cabeçalho de uma seção."""
    blank(out)
    print(bold(text.upper()), file=out)


def field(label: str, value: str, *, indent: int = 2, width: int = 26,
          note: str | None = None, out: TextIO = sys.stdout) -> None:
    """Exibe campo no formato rótulo -> valor com preenchimento pontilhado."""
    pad = " " * indent
    dots = "." * max(1, width - len(label) - 1)
    line = f"{pad}{label} {dim(dots)} {value}"
    if note:
        line += f"   {dim('(' + note + ')')}"
    print(line, file=out)


def bullet(text: str, *, indent: int = 2, marker: str = "•",
           out: TextIO = sys.stdout) -> None:
    print(f"{' ' * indent}{dim(marker)} {text}", file=out)


def explain(text: str, *, indent: int = 2, out: TextIO = sys.stdout) -> None:
    """Exibe texto informativo com quebra automática de linha."""
    pad = " " * indent
    for line in textwrap.wrap(" ".join(text.split()), width=WIDTH - indent):
        print(dim(f"{pad}{line}"), file=out)


def stage(n: int, total: int, name: str, state: StageState = StageState.RUNNING,
          *, seconds: float | None = None, detail: str | None = None,
          out: TextIO = sys.stdout) -> None:
    """Exibe linha de status para etapa do pipeline."""
    mark, color = _MARKS[state]
    counter = dim(f"[{n}/{total}]")
    label = bold(name) if state is StageState.RUNNING else name
    left = f"{counter} {_paint(mark, color) if color else mark} {label}"

    right = ""
    if seconds is not None:
        right = _format_seconds(seconds)
    elif state is not StageState.RUNNING:
        right = _LABELS[state]

    if right:
        gap = max(1, WIDTH - _visible_len(left) - len(right) - 1)
        print(f"{left} {dim('.' * gap)} {right}", file=out)
    else:
        print(left, file=out)

    if detail:
        print(dim(f"        {detail}"), file=out)


def step(text: str, *, indent: int = 8, out: TextIO = sys.stdout) -> None:
    """Exibe linha de progresso secundária."""
    print(f"{' ' * indent}{text}", file=out)


def table(headers: Sequence[str], rows: Iterable[Sequence[str]],
          *, indent: int = 2, align_right: Sequence[int] = (),
          out: TextIO = sys.stdout) -> None:
    """Renderiza tabela alinhada por colunas."""
    rows = [list(r) for r in rows]
    if not rows:
        return

    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(str(cell)))

    def render(cells: Sequence[str], style=lambda s: s) -> str:
        parts = []
        for i, cell in enumerate(cells):
            text = str(cell)
            pad = widths[i] - _visible_len(text)
            parts.append(" " * pad + text if i in align_right else text + " " * pad)
        return " " * indent + style("  ".join(parts).rstrip())

    print(render(headers, bold), file=out)
    print(" " * indent + dim("─" * (sum(widths) + 2 * (len(widths) - 1))), file=out)
    for row in rows:
        print(render(row), file=out)


def check(label: str, state: StageState, detail: str = "",
          *, hint: str | None = None, width: int = 20,
          out: TextIO = sys.stdout) -> None:
    """Exibe linha de diagnóstico de ambiente."""
    mark, color = _MARKS[state]
    symbol = _paint(mark, color) if color else mark
    print(f"  {symbol} {label:<{width}} {dim(detail)}", file=out)
    if hint:
        print(f"    {' ' * width} {warning('→')} {hint}", file=out)


# --------------------------------------------------------------- mensagens


def note(text: str, out: TextIO = sys.stdout) -> None:
    print(f"  {accent('i')} {text}", file=out)


def _sync() -> None:
    sys.stdout.flush()


def warn(text: str, hint: str | None = None) -> None:
    """Emite mensagem de aviso em stderr."""
    _sync()
    print(f"\n  {warning('aviso')}  {text}", file=sys.stderr)
    if hint:
        print(f"         {dim('→')} {hint}", file=sys.stderr)
    sys.stderr.flush()


def error(message: str, *, cause: str | None = None, action: str | None = None) -> None:
    """Emite mensagem de erro em stderr com causa e ação sugerida."""
    _sync()
    print(file=sys.stderr)
    print(f"  {failure('erro')}  {bold(message)}", file=sys.stderr)
    if cause:
        for line in textwrap.wrap(cause, width=WIDTH - 8):
            print(f"        {dim(line)}", file=sys.stderr)
    if action:
        print(file=sys.stderr)
        print(f"  {accent('ação')}  {action}", file=sys.stderr)
    print(file=sys.stderr)
    sys.stderr.flush()


def done(message: str, *, hint: str | None = None, out: TextIO = sys.stdout) -> None:
    """Exibe mensagem de conclusão bem-sucedida."""
    blank(out)
    print(f"  {success('✓')} {bold(message)}", file=out)
    if hint:
        print(f"    {dim(hint)}", file=out)


def command(cmd: str, *, label: str = "execute", out: TextIO = sys.stdout) -> None:
    """Exibe comando sugerido para execução."""
    print(f"  {dim(label)}", file=out)
    print(f"    {accent(cmd)}", file=out)


# ------------------------------------------------------------ formatadores


def _format_seconds(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.0f} min"
    return f"{s:.0f} s"


def format_params(n: int) -> str:
    """Formata contagem de parâmetros com sufixos k, M ou B."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} k"
    return str(n)


def format_bytes(b: float) -> str:
    """Formata valor em bytes para KB, MB ou GB."""
    for unit, size in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if b >= size:
            return f"{b / size:.2f} {unit}"
    return f"{b:.0f} B"


format_seconds = _format_seconds

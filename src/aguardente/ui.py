"""Formatação e apresentação de interface no terminal."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import threading
import time
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


def _stream(out: "TextIO | None") -> "TextIO":
    """Resolve o destino na chamada.

    Fixar `sys.stdout` como valor padrão no import congela o objeto que existia
    naquele momento, o que impede a captura da saída em teste.
    """
    return out if out is not None else sys.stdout


def blank(out: TextIO | None = None) -> None:
    print(file=_stream(out))


def rule(char: str = "─", out: TextIO | None = None) -> None:
    """Imprime linha divisória horizontal."""
    print(dim(char * WIDTH), file=_stream(out))


def title(text: str, subtitle: str | None = None, out: TextIO | None = None) -> None:
    """Exibe o cabeçalho principal do comando."""
    blank(out)
    print(bold(text), file=_stream(out))
    if subtitle:
        print(dim(subtitle), file=_stream(out))
    rule(out=out)


def header(text: str, out: TextIO | None = None) -> None:
    """Exibe o cabeçalho de uma seção."""
    blank(out)
    print(bold(text.upper()), file=_stream(out))


def field(label: str, value: str, *, indent: int = 2, width: int = 26,
          note: str | None = None, out: TextIO | None = None) -> None:
    """Exibe campo no formato rótulo -> valor com preenchimento pontilhado."""
    pad = " " * indent
    dots = "." * max(1, width - len(label) - 1)
    line = f"{pad}{label} {dim(dots)} {value}"
    if note:
        line += f"   {dim('(' + note + ')')}"
    print(line, file=_stream(out))


def bullet(text: str, *, indent: int = 2, marker: str = "•",
           out: TextIO | None = None) -> None:
    print(f"{' ' * indent}{dim(marker)} {text}", file=_stream(out))


def explain(text: str, *, indent: int = 2, out: TextIO | None = None) -> None:
    """Exibe texto informativo com quebra automática de linha."""
    pad = " " * indent
    for line in textwrap.wrap(" ".join(text.split()), width=WIDTH - indent):
        print(dim(f"{pad}{line}"), file=_stream(out))


def stage(n: int, total: int, name: str, state: StageState = StageState.RUNNING,
          *, seconds: float | None = None, detail: str | None = None,
          out: TextIO | None = None) -> None:
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
        print(f"{left} {dim('.' * gap)} {right}", file=_stream(out))
    else:
        print(left, file=_stream(out))

    if detail:
        print(dim(f"        {detail}"), file=_stream(out))


def step(text: str, *, indent: int = 8, out: TextIO | None = None) -> None:
    """Exibe linha de progresso secundária."""
    print(f"{' ' * indent}{text}", file=_stream(out))


def table(headers: Sequence[str], rows: Iterable[Sequence[str]],
          *, indent: int = 2, align_right: Sequence[int] = (),
          out: TextIO | None = None) -> None:
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

    print(render(headers, bold), file=_stream(out))
    print(" " * indent + dim("─" * (sum(widths) + 2 * (len(widths) - 1))), file=_stream(out))
    for row in rows:
        print(render(row), file=_stream(out))


def check(label: str, state: StageState, detail: str = "",
          *, hint: str | None = None, width: int = 20,
          out: TextIO | None = None) -> None:
    """Exibe linha de diagnóstico de ambiente."""
    mark, color = _MARKS[state]
    symbol = _paint(mark, color) if color else mark
    print(f"  {symbol} {label:<{width}} {dim(detail)}", file=_stream(out))
    if hint:
        print(f"    {' ' * width} {warning('→')} {hint}", file=_stream(out))


# --------------------------------------------------------------- animação

# Animação exige um terminal de verdade: num pipe, num log ou sob `--json`, o
# retorno de carro vira lixo no arquivo. A variável permite desligar à mão.
ANIM_ENV = "AGUARDENTE_NO_ANIM"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BAR_FULL, BAR_HEAD, BAR_EMPTY = "━", "╸", "─"
# Altura crescente para os blocos atingidos e ponto para os demais: a cor
# reforça o nível, mas nunca é o único canal — em pipe ou sem cor a leitura
# continua correta.
METER_HEIGHTS = "▁▃▅▇"
METER_EMPTY = "·"


def animations_enabled() -> bool:
    """Indica se a saída comporta atualização in-place."""
    if os.environ.get(ANIM_ENV):
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    return sys.stdout.isatty()


def _clip(text: str, limit: int) -> str:
    """Corta texto puro para caber na largura, sem quebrar a linha do terminal."""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _erase(stream: TextIO) -> None:
    stream.write("\r\033[K")
    stream.flush()


class Spinner:
    """Indicador de atividade para operações sem progresso mensurável.

    Enquanto o spinner está ativo ele é o dono da linha atual: nada mais deve
    escrever na saída até o `stop`, ou as duas escritas se embaralham.
    """

    def __init__(self, text: str = "", *, indent: int = 6, interval: float = 0.08,
                 out: TextIO | None = None) -> None:
        self._text = text
        self._indent = indent
        self._interval = interval
        self._out = out
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def update(self, text: str) -> None:
        """Troca a mensagem sem interromper a animação."""
        with self._lock:
            self._text = text

    def _draw(self, frame: str) -> None:
        with self._lock:
            texto = self._text
        largura = WIDTH - self._indent - 2
        linha = f"{' ' * self._indent}{accent(frame)} {_clip(texto, largura)}"
        stream = _stream(self._out)
        stream.write("\r\033[K" + linha)
        stream.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(self._interval):
            i += 1
            self._draw(SPINNER_FRAMES[i % len(SPINNER_FRAMES)])

    def __enter__(self) -> Spinner:
        if animations_enabled():
            self._draw(SPINNER_FRAMES[0])
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        elif self._text:
            step(self._text, indent=self._indent, out=self._out)
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.stop(ok=exc_type is None)

    def stop(self, *, ok: bool = True, text: str | None = None) -> None:
        """Encerra a animação e deixa uma linha final estática no lugar."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1)
            self._thread = None
            _erase(_stream(self._out))
            final = self._text if text is None else text
            if final:
                mark, color = _MARKS[StageState.OK if ok else StageState.FAILED]
                print(f"{' ' * self._indent}{_paint(mark, color)} {final}",
                      file=_stream(self._out))
        elif text:
            step(text, indent=self._indent, out=self._out)


class Progress:
    """Barra de progresso que se atualiza no lugar, com estimativa de término.

    Sem terminal interativo a barra degrada para marcos textuais a cada 10%,
    o que mantém o log de uma execução longa legível sem virar uma parede de
    linhas repetidas.
    """

    def __init__(self, total: int, *, label: str = "", width: int = 22,
                 indent: int = 6, out: TextIO | None = None) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.indent = indent
        self._out = out
        self._current = 0
        self._started = time.monotonic()
        self._last_mark = -1
        self._animated = animations_enabled()

    @property
    def current(self) -> int:
        return self._current

    @property
    def fraction(self) -> float:
        return min(1.0, self._current / self.total)

    def bar(self) -> str:
        """Desenha a barra: preenchido, cabeça e trilho restante."""
        cheios = int(self.fraction * self.width)
        if cheios >= self.width:
            return success(BAR_FULL * self.width)
        corpo = accent(BAR_FULL * cheios + BAR_HEAD)
        return corpo + dim(BAR_EMPTY * (self.width - cheios - 1))

    def _eta(self) -> str:
        decorrido = time.monotonic() - self._started
        if self._current <= 0 or decorrido < 1:
            return ""
        restante = decorrido / self._current * (self.total - self._current)
        return f" · resta {_format_seconds(restante)}" if restante >= 1 else ""

    def advance(self, n: int = 1, *, suffix: str = "") -> None:
        self.update(self._current + n, suffix=suffix)

    def update(self, current: int, *, suffix: str = "") -> None:
        self._current = max(0, min(self.total, int(current)))
        pct = int(self.fraction * 100)
        if self._animated:
            cauda = self._eta() + (f" · {suffix}" if suffix else "")
            texto = (f"{' ' * self.indent}{self.label} {self.bar()} "
                     f"{pct:3d}%{dim(cauda) if cauda else ''}")
            stream = _stream(self._out)
            stream.write("\r\033[K" + texto)
            stream.flush()
            return
        marco = pct // 10
        if marco > self._last_mark:
            self._last_mark = marco
            step(f"{self.label} {pct:3d}% ({self._current}/{self.total}){' ' + suffix if suffix else ''}",
                 indent=self.indent, out=self._out)

    def done(self, *, suffix: str = "") -> None:
        """Fecha a barra deixando uma linha final estática."""
        decorrido = _format_seconds(time.monotonic() - self._started)
        resumo = suffix or f"{self.total} unidades"
        if self._animated:
            _erase(_stream(self._out))
            mark, color = _MARKS[StageState.OK]
            print(f"{' ' * self.indent}{_paint(mark, color)} {self.label} "
                  f"{dim(f'· {resumo} em {decorrido}')}", file=_stream(self._out))
        else:
            step(f"{self.label} concluído: {resumo} em {decorrido}",
                 indent=self.indent, out=self._out)


def meter(level: int, *, total: int = 4) -> str:
    """Medidor em blocos de altura crescente, com os níveis acima como pontos."""
    partes = []
    for i in range(total):
        if i <= level:
            partes.append(accent(METER_HEIGHTS[min(i, len(METER_HEIGHTS) - 1)]))
        else:
            partes.append(dim(METER_EMPTY))
    return "".join(partes)


def meter_line(level: int, *, total: int = 4, label: str = "", indent: int = 2,
               animate: bool = True, out: TextIO | None = None) -> None:
    """Exibe o medidor preenchendo bloco a bloco até o nível escolhido."""
    stream = _stream(out)
    if animate and animations_enabled():
        for passo in range(level + 1):
            stream.write(f"\r\033[K{' ' * indent}{meter(passo, total=total)}  {label}")
            stream.flush()
            time.sleep(0.09)
        stream.write("\n")
        stream.flush()
        return
    print(f"{' ' * indent}{meter(level, total=total)}  {label}", file=stream)


# --------------------------------------------------------------- mensagens


def note(text: str, out: TextIO | None = None) -> None:
    print(f"  {accent('i')} {text}", file=_stream(out))


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


def done(message: str, *, hint: str | None = None, out: TextIO | None = None) -> None:
    """Exibe mensagem de conclusão bem-sucedida."""
    blank(out)
    print(f"  {success('✓')} {bold(message)}", file=_stream(out))
    if hint:
        print(f"    {dim(hint)}", file=_stream(out))


def command(cmd: str, *, label: str = "execute", out: TextIO | None = None) -> None:
    """Exibe comando sugerido para execução."""
    print(f"  {dim(label)}", file=_stream(out))
    print(f"    {accent(cmd)}", file=_stream(out))


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

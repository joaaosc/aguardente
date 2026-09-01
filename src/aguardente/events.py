"""Canal de eventos estruturado, paralelo à saída legível.

Uma linha JSON por evento. É o que permite retomar uma execução e, mais tarde,
alimentar uma interface gráfica sem parsear texto formatado.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class EventLog:
    """Escreve NDJSON num arquivo e, opcionalmente, num stream."""

    path: Path | None = None
    stream: TextIO | None = None
    _fh: Any = field(default=None, init=False, repr=False)

    def __enter__(self) -> EventLog:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "ev": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        if self._fh:
            self._fh.write(line)
            self._fh.flush()
        if self.stream:
            self.stream.write(line)
            self.stream.flush()

    # Vocabulário de eventos — nomes fixos para que consumidores não adivinhem.
    def plan(self, stages: list[dict[str, Any]]) -> None:
        self.emit("plan", stages=stages)

    def stage_start(self, sid: str, n: int, of: int, rationale: str = "") -> None:
        self.emit("stage.start", id=sid, n=n, of=of, rationale=rationale)

    def stage_end(self, sid: str, ok: bool, ms: int) -> None:
        self.emit("stage.end", id=sid, ok=ok, ms=ms)

    def progress(self, sid: str, current: int, total: int | None = None) -> None:
        self.emit("progress", id=sid, cur=current, tot=total)

    def metric(self, sid: str, key: str, value: float, unit: str = "") -> None:
        self.emit("metric", id=sid, k=key, v=value, unit=unit)

    def log(self, sid: str, message: str, level: str = "info") -> None:
        self.emit("log", id=sid, msg=message, level=level)

    def error(self, sid: str, message: str, hint: str | None = None) -> None:
        self.emit("error", id=sid, msg=message, hint=hint)


def null_log() -> EventLog:
    return EventLog(path=None, stream=None)


def stdout_log() -> EventLog:
    return EventLog(path=None, stream=sys.stdout)

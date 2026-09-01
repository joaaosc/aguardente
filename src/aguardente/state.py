"""Estado de uma execução, para retomada.

Um pipeline que baixa 7,5 GB e treina por horas não pode perder tudo num
Ctrl-C ou numa queda de rede. Cada etapa grava o que produziu; ao retomar, o
que já está pronto é pulado.

O estado é um JSON simples e legível — dá para inspecionar e editar à mão
quando algo dá errado, que é exatamente quando se precisa disso.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

STATE_FILE = "state.json"
STATE_VERSION = 1


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    status: StageStatus = StageStatus.PENDING
    started: float | None = None
    ended: float | None = None
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def seconds(self) -> float | None:
        if self.started and self.ended:
            return self.ended - self.started
        return None

    @property
    def done(self) -> bool:
        return self.status in (StageStatus.OK, StageStatus.SKIPPED)


@dataclass
class RunState:
    run_dir: Path
    model: str = ""
    target_params: int | None = None
    created: float = field(default_factory=time.time)
    version: int = STATE_VERSION
    stages: dict[str, StageState] = field(default_factory=dict)

    # ---------------------------------------------------------------- io

    @property
    def path(self) -> Path:
        return self.run_dir / STATE_FILE

    @classmethod
    def load_or_create(cls, run_dir: str | Path, *, model: str = "",
                       target_params: int | None = None) -> RunState:
        d = Path(run_dir).expanduser()
        p = d / STATE_FILE
        if p.is_file():
            raw = json.loads(p.read_text())
            state = cls(
                run_dir=d,
                model=raw.get("model", model),
                target_params=raw.get("target_params", target_params),
                created=raw.get("created", time.time()),
                version=raw.get("version", STATE_VERSION),
                stages={k: StageState(status=StageStatus(v.get("status", "pending")),
                                      started=v.get("started"), ended=v.get("ended"),
                                      error=v.get("error"), outputs=v.get("outputs", {}),
                                      metrics=v.get("metrics", {}))
                        for k, v in raw.get("stages", {}).items()},
            )
            # Uma etapa marcada "running" num estado carregado do disco é órfã:
            # o processo anterior morreu no meio. Volta para pendente.
            for s in state.stages.values():
                if s.status is StageStatus.RUNNING:
                    s.status = StageStatus.PENDING
            return state

        d.mkdir(parents=True, exist_ok=True)
        state = cls(run_dir=d, model=model, target_params=target_params)
        state.save()
        return state

    def save(self) -> None:
        payload = {
            "version": self.version,
            "model": self.model,
            "target_params": self.target_params,
            "created": self.created,
            "stages": {k: {**asdict(v), "status": v.status.value}
                       for k, v in self.stages.items()},
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.replace(self.path)     # atômico: um Ctrl-C não deixa o estado corrompido

    # ---------------------------------------------------------------- etapas

    def stage(self, name: str) -> StageState:
        return self.stages.setdefault(name, StageState())

    def is_done(self, name: str, *, require: Sequence[str] = ()) -> bool:
        """Concluída **e** com as saídas ainda no disco.

        A segunda condição importa: alguém pode ter apagado a pasta entre
        execuções, e confiar só no JSON produziria um erro confuso lá adiante.
        """
        s = self.stage(name)
        if not s.done:
            return False
        for key in require:
            p = s.outputs.get(key)
            if not p or not Path(p).exists():
                return False
        return True

    def begin(self, name: str) -> StageState:
        s = self.stage(name)
        s.status = StageStatus.RUNNING
        s.started = time.time()
        s.error = None
        self.save()
        return s

    def finish(self, name: str, *, outputs: dict[str, Any] | None = None,
               metrics: dict[str, float] | None = None) -> None:
        s = self.stage(name)
        s.status = StageStatus.OK
        s.ended = time.time()
        if outputs:
            s.outputs.update({k: str(v) for k, v in outputs.items()})
        if metrics:
            s.metrics.update(metrics)
        self.save()

    def fail(self, name: str, error: str) -> None:
        s = self.stage(name)
        s.status = StageStatus.FAILED
        s.ended = time.time()
        s.error = error
        self.save()

    def skip(self, name: str, reason: str = "") -> None:
        s = self.stage(name)
        s.status = StageStatus.SKIPPED
        s.ended = time.time()
        s.error = reason or None
        self.save()

    def output(self, name: str, key: str) -> Path | None:
        p = self.stage(name).outputs.get(key)
        return Path(p) if p else None

    def summary(self) -> list[tuple[str, StageStatus, float | None]]:
        return [(k, v.status, v.seconds) for k, v in self.stages.items()]

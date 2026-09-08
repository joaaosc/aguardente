"""Gerenciamento de estado de execução para suporte a retomada."""

from __future__ import annotations

import json
import os
import socket
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Sequence

from .errors import StateMismatch

STATE_FILE = "state.json"
STATE_VERSION = 1

# Chave em que cada etapa grava a impressão digital dos parâmetros que a geraram.
FINGERPRINT = "fingerprint"

LOCK_FILE = "run.lock"
# Um lock mais antigo que isto pertence a um processo que morreu sem limpá-lo.
STALE_LOCK_SECONDS = 12 * 3600
ARTIFACT_DIRS = ("teacher", "teacher-text", "pruned", "logits", "student", "bundle", "ckpt", "data", "dry-run")


def discard_run_artifacts(out_dir: Path) -> list[str]:
    """Called with run_lock held; keep the lock and user-owned files intact."""
    removed = []
    for name in (STATE_FILE, "quality.json", "reconstruction.json", "runtime-validation.json"):
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    for name in ARTIFACT_DIRS:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(f"{name}/")
    return removed


def _process_alive(pid: int) -> bool:
    """Indica se o processo existe; sinal 0 apenas consulta, não interrompe."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # existe, mas pertence a outro usuário
        return True
    return True


def _lock_owner(lock: Path) -> dict[str, Any] | None:
    try:
        dados = json.loads(lock.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return dados if isinstance(dados, dict) and {"pid", "host", "at"} <= dados.keys() else None


def run_is_active(run_dir: str | Path) -> bool:
    owner = _lock_owner(Path(run_dir) / LOCK_FILE)
    return bool(owner and (owner["host"] != socket.gethostname() or _process_alive(int(owner["pid"]))))


def _lock_is_stale(dono: dict[str, Any], stale_after: float) -> bool:
    """Treinos longos não perdem o lock enquanto o processo estiver vivo."""
    # A verificação por pid só vale na mesma máquina: em volume compartilhado o
    # número poderia coincidir com um processo local sem qualquer relação.
    if dono.get("host") != socket.gethostname():
        return False
    return not _process_alive(int(dono.get("pid") or 0))


@contextmanager
def run_lock(run_dir: str | Path, *,
             stale_after: float = STALE_LOCK_SECONDS) -> Iterator[Path]:
    """Exclusão mútua consultiva entre execuções sobre o mesmo diretório."""
    d = Path(run_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    lock = d / LOCK_FILE
    marca = json.dumps({"pid": os.getpid(), "at": time.time(),
                        "host": socket.gethostname()})

    for tentativa in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, marca.encode())
            os.close(fd)
            break
        except FileExistsError:
            dono = _lock_owner(lock)
            incomplete_stale = dono is None and time.time() - lock.stat().st_mtime > min(60, stale_after)
            if tentativa == 1 and (incomplete_stale or (dono is not None and _lock_is_stale(dono, stale_after))):
                lock.unlink(missing_ok=True)
                continue
            raise StateMismatch(
                f"outra execução já está usando {d} "
                f"(pid {dono.get('pid') if dono else '?'})",
                hint=f"Aguarde a conclusão, ou remova {lock} se o processo não existe mais.",
            ) from None
    try:
        yield lock
    finally:
        dono = _lock_owner(lock)
        if dono and dono.get("pid") == os.getpid():
            lock.unlink(missing_ok=True)


def _require_same(rotulo: str, gravado: Any, pedido: Any, run_dir: Path) -> None:
    """Recusa reaproveitar um diretório cujo estado foi criado com outro parâmetro.

    O argumento vazio (ou ausente) herda o valor gravado — é o caso da retomada
    normal. Só há divergência quando ambos existem e diferem, e nesse caso a
    execução para: continuar produziria um artefato que não corresponde ao comando.
    """
    if not pedido or not gravado or pedido == gravado:
        return
    raise StateMismatch(
        f"o diretório {run_dir} pertence a outra execução — "
        f"{rotulo} gravado: {gravado}; solicitado: {pedido}",
        hint="Use outro diretório com -o, ou --restart para descartar o estado e os "
             "artefatos desta pasta antes de recomeçar.",
    )


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
    """Estado persistido da execução do pipeline."""

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
                       target_params: int | None = None,
                       restart: bool = False, reset_interrupted: bool = True) -> RunState:
        d = Path(run_dir).expanduser()
        p = d / STATE_FILE
        if p.is_file() and not restart:
            raw = cls._read(p)
            gravado_modelo = str(raw.get("model") or "")
            gravado_alvo = raw.get("target_params")
            _require_same("modelo", gravado_modelo, model, d)
            _require_same("alvo de parâmetros", gravado_alvo, target_params, d)
            state = cls(
                run_dir=d,
                model=gravado_modelo or model,
                target_params=gravado_alvo if gravado_alvo is not None else target_params,
                created=raw.get("created", time.time()),
                version=raw.get("version", STATE_VERSION),
                stages={k: StageState(status=StageStatus(v.get("status", "pending")),
                                      started=v.get("started"), ended=v.get("ended"),
                                      error=v.get("error"), outputs=v.get("outputs", {}),
                                      metrics=v.get("metrics", {}))
                        for k, v in raw.get("stages", {}).items()},
            )
            # Reseta etapas interrompidas que ficaram salvas como "running"
            for s in state.stages.values():
                if reset_interrupted and s.status is StageStatus.RUNNING:
                    s.status = StageStatus.PENDING
            return state

        # O arquivo não é gravado aqui: ele materializa na primeira etapa que
        # começa. Uma execução que morre antes disso — identificador errado,
        # modelo grande demais para a RAM, ambiente incompleto — não deixa a
        # pasta reservada em nome de um modelo que nunca chegou a ser baixado,
        # o que obrigaria `--restart` para tentar de novo com o nome correto.
        d.mkdir(parents=True, exist_ok=True)
        return cls(run_dir=d, model=model, target_params=target_params)

    @staticmethod
    def _read(p: Path) -> dict[str, Any]:
        """Lê o estado gravado, traduzindo arquivo corrompido em erro acionável."""
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise StateMismatch(
                f"{p} está corrompido e não pôde ser lido: {e}",
                hint="Remova o arquivo para reiniciar a execução, ou use outro diretório com -o.",
            ) from e
        except OSError as e:
            raise StateMismatch(f"não foi possível ler {p}: {e}",
                                hint="Verifique as permissões do diretório de execução.") from e
        if not isinstance(raw, dict):
            raise StateMismatch(f"{p} não contém um objeto JSON válido",
                                hint="Remova o arquivo para reiniciar a execução.")
        return raw

    def save(self) -> None:
        """Grava o estado em disco de forma atômica."""
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
        tmp.replace(self.path)

    # ---------------------------------------------------------------- etapas

    def stage(self, name: str) -> StageState:
        return self.stages.setdefault(name, StageState())

    def is_done(self, name: str, *, require: Sequence[str] = (),
                fingerprint: str | None = None) -> bool:
        """Verifica se a etapa foi concluída, se as saídas existem e se os parâmetros batem."""
        s = self.stage(name)
        if not s.done:
            return False
        for key in require:
            p = s.outputs.get(key)
            if not p or not Path(p).exists():
                return False
        if fingerprint is not None:
            gravado = s.outputs.get(FINGERPRINT)
            # Estado sem impressão digital vem de uma versão anterior: é aproveitado,
            # e o chamador avisa. Divergência explícita invalida a etapa.
            if gravado is not None and gravado != fingerprint:
                return False
        return True

    def fingerprint_of(self, name: str) -> str | None:
        """Impressão digital gravada pela etapa, ou None se ela nunca gravou uma."""
        return self.stage(name).outputs.get(FINGERPRINT)

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

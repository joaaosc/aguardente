"""Treinamento de recuperação via destilação de logits."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .loss import kd_loss
from .teacher import TeacherLogits

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Hiperparâmetros e configurações do treino de recuperação."""

    epochs: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    alpha: float = 0.9
    temperature: float = 2.0
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.03
    gradient_checkpointing: bool = True
    eval_every: int = 0
    plateau_threshold: float = 0.005
    plateau_patience: int = 2
    max_seconds: float | None = None


# Janela máxima de histórico de perdas mantida em memória
_MAX_LOSS_HISTORY = 2000


@dataclass
class RecoveryResult:
    steps: int = 0
    epochs_completed: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    evals: list[tuple[int, float]] = field(default_factory=list)
    stopped_by: str = "epochs"

    @property
    def best_eval(self) -> float | None:
        return min((v for _, v in self.evals), default=None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _build_optimizer(model: Any, cfg: RecoveryConfig) -> Any:
    import torch

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
    )


def _lr_at(step: int, total: int, cfg: RecoveryConfig) -> float:
    """Calcula a taxa de aprendizado com warmup linear e decaimento por cosseno."""
    warmup = max(1, int(total * cfg.warmup_fraction))
    if step < warmup:
        return cfg.learning_rate * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def recover(
    student: Any,
    logits: TeacherLogits,
    cfg: RecoveryConfig | None = None,
    *,
    device: str | None = None,
    evaluate: Callable[[], float] | None = None,
    on_step: Callable[[int, float], None] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> RecoveryResult:
    """Executa o treinamento de recuperação do student contra os logits do teacher."""
    import torch

    cfg = cfg or RecoveryConfig()
    dev = device or str(next(student.parameters()).device)
    result = RecoveryResult()

    if cfg.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "config"):
            student.config.use_cache = False

    optimizer = _build_optimizer(student, cfg)
    total_steps = max(1, cfg.epochs * logits.shards // cfg.grad_accum)
    eval_every = cfg.eval_every or max(1, logits.shards // 2)

    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    stale = 0
    started = time.perf_counter()
    student.train()

    try:
        for epoch in range(cfg.epochs):
            for i, batch in enumerate(logits.batches(device=dev)):
                lr = _lr_at(result.steps, total_steps, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                out = student(input_ids=batch["input_ids"])
                loss = kd_loss(
                    out.logits, batch["values"], batch["indices"],
                    labels=batch["input_ids"],
                    alpha=cfg.alpha, temperature=cfg.temperature,
                )
                (loss / cfg.grad_accum).backward()

                if (i + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    result.steps += 1

                value = float(loss.detach())
                result.losses.append(value)
                if len(result.losses) > _MAX_LOSS_HISTORY:
                    del result.losses[:len(result.losses) - _MAX_LOSS_HISTORY]
                if on_step:
                    on_step(result.steps, value)

                if cfg.max_seconds and time.perf_counter() - started > cfg.max_seconds:
                    result.stopped_by = "time"
                    return _finish(result, started, student, ckpt)

                if evaluate and (i + 1) % eval_every == 0:
                    student.eval()
                    metric = float(evaluate())
                    student.train()
                    result.evals.append((result.steps, metric))

                    gain = (best - metric) / best if math.isfinite(best) else 1.0
                    if gain > cfg.plateau_threshold:
                        best, stale = metric, 0
                        if ckpt:
                            _save_checkpoint(student, ckpt, result.steps, metric)
                    else:
                        stale += 1
                        if stale >= cfg.plateau_patience:
                            result.stopped_by = "plateau"
                            return _finish(result, started, student, ckpt)

            result.epochs_completed = epoch + 1
    except KeyboardInterrupt:
        result.stopped_by = "interrupted"

    return _finish(result, started, student, ckpt)


def _finish(result: RecoveryResult, started: float, student: Any,
            ckpt: Path | None) -> RecoveryResult:
    result.seconds = time.perf_counter() - started
    student.eval()
    if hasattr(student, "config"):
        student.config.use_cache = True
    if ckpt:
        (ckpt / "recovery.json").write_text(result.to_json())
    return result


def _save_checkpoint(student: Any, ckpt: Path, step: int, metric: float) -> None:
    """Grava o melhor checkpoint em disco de forma atômica."""
    import torch

    destino = ckpt / "best.pt"
    temporario = ckpt / "best.pt.tmp"
    try:
        torch.save(
            {"step": step, "metric": metric,
             "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()}},
            temporario,
        )
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)

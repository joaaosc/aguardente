"""Treinamento de recuperação via destilação de logits."""

from __future__ import annotations

import json
import math
import random
import tempfile
import time
import signal
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .loss import kd_loss
from .teacher import TeacherLogits
from ..errors import AguardenteError

@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Hiperparâmetros e configurações do treino de recuperação."""

    epochs: int = 2
    seed: int = 42
    shuffle: bool = True
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
    # Passos de otimizador entre checkpoints completos para retomada.
    checkpoint_every: int = 50


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
    # Passos já feitos ao retomar de um checkpoint; 0 quando o treino começou
    # do zero. Só para relatório — não entra no cálculo do que falta rodar.
    resumed_from: int = 0
    tokens_seen: int = 0
    best_step: int | None = None

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


def _lr_at(step: int, total: int, cfg: RecoveryConfig, *, min_lr_fraction: float = 0.1) -> float:
    """Calcula a taxa de aprendizado com warmup linear e decaimento por cosseno até um piso mínimo."""
    warmup = max(1, int(total * cfg.warmup_fraction))
    if step < warmup:
        return cfg.learning_rate * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    min_lr = cfg.learning_rate * min_lr_fraction
    return min_lr + (cfg.learning_rate - min_lr) * cosine


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
    """Executa o treinamento de recuperação do student contra os logits do teacher.

    Quando `checkpoint_dir` já contém um checkpoint de uma execução anterior,
    a retomada é automática: pesos, estado do otimizador e a posição exata no
    fluxo de lotes voltam de onde pararam, sem repetir trabalho já feito.
    """
    import torch

    cfg = cfg or RecoveryConfig()
    if cfg.epochs < 1 or cfg.grad_accum < 1 or logits.shards < 1:
        raise ValueError("epochs, grad_accum and logits.shards must be positive")
    if cfg.learning_rate <= 0 or not math.isfinite(cfg.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if cfg.plateau_patience < 1:
        raise ValueError("plateau_patience must be positive")
    dev = device or str(next(student.parameters()).device)
    result = RecoveryResult()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if hasattr(student, "config"):
        student.config.use_cache = False
    if cfg.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    if hasattr(student, "float"):
        student.float()
    if logits.temperature != cfg.temperature:
        raise AguardenteError("temperatura dos logits difere da temperatura do treino; regenere os logits")
    optimizer = _build_optimizer(student, cfg)
    steps_per_epoch = math.ceil(logits.shards / cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    eval_every = cfg.eval_every or max(1, steps_per_epoch // 4)
    # Best weights live on disk even for API callers without persistent checkpoints.
    temporary = tempfile.TemporaryDirectory(prefix="aguardente-best-") if evaluate and not checkpoint_dir else None
    ckpt = Path(checkpoint_dir or temporary.name) if checkpoint_dir or temporary else None
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)
    best = plateau_best = float("inf")
    stale = lotes_feitos = 0
    if ckpt and (ckpt / "last.pt").is_file():
        estado = _load_checkpoint(ckpt, student, optimizer, device=dev)
        result.steps = result.resumed_from = estado["step"]
        lotes_feitos = estado["lotes_feitos"]
        best = estado["best"]
        plateau_best = estado.get("plateau_best", best)
        stale = estado.get("stale", 0)
        result.tokens_seen = estado.get("tokens_seen", 0)
        result.evals = estado.get("evals", [])
        result.best_step = estado.get("best_step")
    started = time.perf_counter()
    student.train()
    pending = pending_tokens = 0
    window_rng = None

    def checkpoint():
        if ckpt:
            _save_checkpoint_resumavel(student, optimizer, ckpt, result.steps,
                lotes_feitos, best, stale, plateau_best=plateau_best,
                tokens_seen=result.tokens_seen, evals=result.evals, best_step=result.best_step)

    def assess():
        nonlocal best, plateau_best, stale
        if not evaluate:
            return False
        student.eval()
        try:
            metric = float(evaluate())
        finally:
            student.train()
        if not math.isfinite(metric) or metric <= 0:
            raise AguardenteError(f"métrica de validação inválida: {metric}")
        result.evals.append((result.steps, metric))
        # Saving the actual minimum is independent from the patience threshold.
        if metric < best:
            best = metric
            result.best_step = result.steps
            if ckpt:
                _save_checkpoint(student, ckpt, result.steps, metric)
        gain = (plateau_best - metric) / plateau_best if math.isfinite(plateau_best) else 1.0
        if gain > cfg.plateau_threshold:
            plateau_best, stale = metric, 0
        else:
            stale += 1
        return stale >= cfg.plateau_patience

    def apply_step():
        nonlocal pending, pending_tokens, window_rng
        # Each microbatch contributes its SUM, then the complete window is
        # normalized by valid causal transitions, including a residual window.
        # Ctrl-C must not checkpoint half of an AdamW update. Finish this small
        # transaction before dispatching the signal to the interrupt handler.
        with _defer_sigint():
            for p in student.parameters():
                if p.grad is not None:
                    p.grad.div_(pending_tokens)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm,
                                          error_if_nonfinite=True)
            lr = _lr_at(result.steps + 1, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            result.steps += 1
            result.tokens_seen += pending_tokens
            pending = pending_tokens = 0
            window_rng = None

    try:
        if evaluate and not result.evals:
            assess()  # The untrained student is a valid best checkpoint too.
        epoca_inicial = lotes_feitos // logits.shards
        for epoch in range(epoca_inicial, cfg.epochs):
            order = list(range(logits.shards))
            if cfg.shuffle:
                random.Random(cfg.seed + epoch).shuffle(order)
            skip = lotes_feitos - epoch * logits.shards if epoch == epoca_inicial else 0
            for i, batch in enumerate(logits.batches(device=dev, start=skip, order=order), start=skip):
                if not pending:
                    window_rng = _capture_rng()
                mask = batch.get("attention_mask")
                ids = batch["input_ids"]
                tokens = int((mask[:, :-1].bool() & mask[:, 1:].bool()).sum()) if mask is not None else ids.shape[0] * (ids.shape[1] - 1)
                if tokens < 1:
                    raise AguardenteError("shard sem transições causais válidas")
                if "logsumexp" not in batch:
                    raise AguardenteError("logits antigos sem massa de probabilidade; regenere a etapa logits")
                out = student(input_ids=ids, **({"attention_mask": mask} if mask is not None else {}))
                loss = kd_loss(out.logits, batch["values"], batch["indices"], labels=ids,
                               mask=mask, alpha=cfg.alpha, temperature=cfg.temperature,
                               teacher_logsumexp=batch["logsumexp"],
                               teacher_tail_values=batch.get("tail_values"),
                               teacher_tail_indices=batch.get("tail_indices"))
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise AguardenteError(f"a perda de destilação divergiu (loss={value}) no passo {result.steps}")
                (loss * tokens).backward()
                lotes_feitos += 1
                pending += 1
                pending_tokens += tokens
                stepped = pending == cfg.grad_accum or i + 1 == logits.shards
                if stepped:
                    apply_step()
                result.losses.append(value)
                del result.losses[:-_MAX_LOSS_HISTORY]
                if stepped:
                    end_epoch = i + 1 == logits.shards
                    if end_epoch:
                        result.epochs_completed = epoch + 1
                    if evaluate and (result.steps % eval_every == 0 or end_epoch):
                        plateau = assess()
                        # Early validation can plateau before the model has
                        # seen half of a large corpus. Finish at least one
                        # complete pass; still save/restore every real minimum.
                        if plateau and 1 <= result.epochs_completed < cfg.epochs:
                            result.stopped_by = "plateau"
                            checkpoint()
                            return _finish(result, started, student, ckpt)
                    if cfg.checkpoint_every and result.steps % cfg.checkpoint_every == 0:
                        checkpoint()
                    if cfg.max_seconds and time.perf_counter() - started > cfg.max_seconds:
                        result.stopped_by = "time"
                        checkpoint()
                        return _finish(result, started, student, ckpt)
                if on_step:
                    on_step(result.steps, value)
        return _finish(result, started, student, ckpt)
    except KeyboardInterrupt:
        # An interrupted backward can leave partial gradients. Replay the whole
        # uncommitted accumulation window, rather than applying corrupt updates.
        optimizer.zero_grad(set_to_none=True)
        lotes_feitos -= pending
        if window_rng is not None:
            _restore_rng(window_rng)
        result.stopped_by = "interrupted"
        checkpoint()
        return _finish(result, started, student, ckpt)
    finally:
        if temporary:
            temporary.cleanup()


def _finish(result: RecoveryResult, started: float, student: Any,
            ckpt: Path | None) -> RecoveryResult:
    result.seconds = time.perf_counter() - started
    student.eval()
    if hasattr(student, "config"):
        student.config.use_cache = True
    # Restaura o melhor modelo registrado durante as avaliações, se houver.
    # Um `best.pt` só pertence a este treino quando ele mesmo o gravou (há
    # avaliações registradas) ou quando esta execução é a continuação de outra
    # (retomada de checkpoint). Um treino que começou do zero e nunca avaliou
    # nada encontraria ali os pesos de uma execução anterior, e carregá-los
    # descartaria em silêncio tudo o que acabou de treinar.
    proprio = bool(result.evals) or result.resumed_from > 0
    if ckpt and proprio and (ckpt / "best.pt").is_file():
        import torch
        dev = next(student.parameters()).device
        blob = torch.load(ckpt / "best.pt", map_location=dev, weights_only=True)
        student.load_state_dict(blob["state_dict"], assign=False)
    _exigir_pesos_finitos(student)
    if ckpt:
        (ckpt / "recovery.json").write_text(result.to_json())
        # Um treino que chegou até aqui por completar as épocas não precisa
        # mais retomar de lugar nenhum; o checkpoint de retomada some, e só
        # sobra `best.pt` — o resultado, não o estado intermediário.
        if result.stopped_by == "epochs":
            (ckpt / "last.pt").unlink(missing_ok=True)
    return result


def _exigir_pesos_finitos(student: Any) -> None:
    """Confere que o treino não deixou infinito ou NaN nos pesos.

    A perda é conferida a cada lote, mas a divergência pode aparecer no passo
    do otimizador que fecha o treino — depois da última perda medida e sem
    nenhuma passada adiante para denunciá-la. Sem esta verificação o student
    corrompido segue para a exportação e vira um bundle que carrega, ocupa uma
    fração do tamanho esperado e não produz nada.
    """
    import torch

    for nome, tensor in student.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise AguardenteError(
                f"a recuperação produziu pesos não finitos em {nome}",
                hint="O treino divergiu no último passo. Reduza --lr, aumente "
                     "--grad-accum, ou use um nível de esforço mais conservador; "
                     "o treino usa FP32 em todos os dispositivos.",
            )


def _save_checkpoint(student: Any, ckpt: Path, step: int, metric: float) -> None:
    """Grava o melhor checkpoint em disco de forma atômica."""
    import torch

    destino = ckpt / "best.pt"
    temporario = ckpt / "best.pt.tmp"
    try:
        torch.save(
            {"step": step, "metric": metric,
             "state_dict": _cpu_state_dict(student)},
            temporario,
        )
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)


def _save_checkpoint_resumavel(student: Any, optimizer: Any, ckpt: Path, step: int,
                               lotes_feitos: int, best: float, stale: int, **extra: Any) -> None:
    """Grava pesos, estado do otimizador e posição — o bastante para retomar do zero.

    Diferente de `_save_checkpoint` (só os pesos, para inspecionar o melhor
    resultado), este arquivo existe para ser recarregado por `_load_checkpoint`
    e continuar o treino exatamente de onde parou, LR e momento do AdamW
    incluídos — sem isso a retomada reiniciaria o otimizador do zero.
    """
    import torch

    destino = ckpt / "last.pt"
    temporario = ckpt / "last.pt.tmp"
    try:
        torch.save(
            {"step": step, "lotes_feitos": lotes_feitos, "best": best, "stale": stale,
             "state_dict": _cpu_state_dict(student),
             "optimizer_state": optimizer.state_dict(), "rng": _capture_rng(), **extra},
            temporario,
        )
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)


def _load_checkpoint(ckpt: Path, student: Any, optimizer: Any, *, device: str) -> dict[str, Any]:
    """Carrega pesos e estado do otimizador de `last.pt`, devolvendo a posição salva.

    `weights_only=True` mantém a mesma postura de segurança usada para os
    shards de logits: o arquivo é local e escrito pelo próprio programa, mas
    tratá-lo como dado não-confiável por padrão não custa nada.
    """
    import torch

    blob = torch.load(ckpt / "last.pt", map_location=device, weights_only=True)
    try:
        student.load_state_dict(blob["state_dict"], assign=False)
    except RuntimeError as e:
        raise AguardenteError(
            f"o checkpoint em {ckpt / 'last.pt'} não corresponde ao student atual",
            hint="O checkpoint foi gravado por um student de outro formato — outro "
                 "alvo de parâmetros, outro modelo de origem. Apague o diretório de "
                 "checkpoints ou use --restart para começar a recuperação do zero.",
        ) from e
    optimizer.load_state_dict(blob["optimizer_state"])
    if "rng" in blob:
        _restore_rng(blob["rng"])
    return {k: v for k, v in blob.items() if k not in {"state_dict", "optimizer_state", "rng"}}


def _cpu_state_dict(student: Any) -> dict[str, Any]:
    # Tied embeddings/lm_head share GPU storage. Moving each key separately to
    # CPU duplicates that large matrix in both host RAM and checkpoint files.
    copies, state = {}, {}
    for key, value in student.state_dict().items():
        identity = (str(value.device), value.data_ptr(), value.dtype, tuple(value.shape), tuple(value.stride()))
        if identity not in copies:
            copies[identity] = value.detach().cpu()
        state[key] = copies[identity]
    return state


def _capture_rng() -> dict[str, Any]:
    import torch
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def _defer_sigint():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    received = []

    def defer(signum, frame):
        received.append((signum, frame))

    signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if received and callable(previous):
        previous(*received[0])

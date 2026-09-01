"""Orquestração das etapas do pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .arch import count_params
from .budget import GB, Budget, Machine
from .errors import AguardenteError, InsufficientResources, PlanImpossible
from .events import EventLog, null_log
from .plan import PrunePlan, plan_for_target
from .probe import ModelProbe, probe
from .state import FINGERPRINT, RunState, run_lock

Reporter = Callable[[str], None]

STAGES = ("fetch", "prune", "logits", "recover", "export")

# Folga exigida sobre a estimativa antes de começar a escrever em disco.
DISK_MARGIN = 1.15

# Parâmetros que determinam a saída de cada etapa. Alterá-los invalida o
# trabalho já gravado: a impressão digital é persistida no estado e conferida
# na retomada, para que uma etapa concluída nunca seja reaproveitada sob uma
# configuração que não a produziu.
FINGERPRINT_FIELDS: dict[str, tuple[str, ...]] = {
    "prune": ("model", "target_params", "calib_batches", "batch_size", "seq_len",
              "calib_dataset", "calib_file"),
    "logits": ("model", "top_k", "seq_len", "batch_size", "logit_batches",
               "calib_dataset", "calib_file"),
    "recover": ("model", "target_params", "top_k", "seq_len", "batch_size",
                "logit_batches", "calib_batches", "calib_dataset", "calib_file",
                "epochs", "lr", "alpha", "temperature", "grad_accum", "no_checkpointing"),
    "export": ("model", "target_params", "platform", "compression",
               "compute_precision", "max_context_length"),
}


def fingerprint(opts: RunOptions, stage: str) -> str:
    """Resumo estável dos parâmetros que determinam a saída de uma etapa."""
    campos = {k: getattr(opts, k) for k in FINGERPRINT_FIELDS[stage]}
    bruto = json.dumps(campos, sort_keys=True, default=str)
    return hashlib.sha256(bruto.encode()).hexdigest()[:16]


class _ProgressFilter:
    """Filtra e formata a saída de progresso do aria2c."""

    _PCT = __import__("re").compile(r"\((\d+)%\)")

    def __init__(self, report: Reporter, *, step: int = 10,
                 bar: _Progress | None = None) -> None:
        self.report = report
        self.step = step
        self.bar = bar
        self.last = -1

    def __call__(self, line: str) -> None:
        m = self._PCT.search(line)
        if not m:
            if line.strip() and not line.startswith(("***", "===", "---", "FILE:")):
                if self.bar is None:
                    self.report(f"             {line.strip()}")
            return
        pct = int(m.group(1))
        if self.bar is not None:
            self.bar.update(pct)
            return
        if pct >= self.last + self.step or pct == 100:
            self.last = pct
            self.report(f"             {pct:3d}%")


@dataclass
class RunOptions:
    """Opções de configuração para execução do pipeline."""

    model: str
    out_dir: Path
    target_params: int | None = None

    # download
    connections: int = 8
    concurrent: int = 4

    # calibração
    calib_batches: int = 32
    batch_size: int = 2
    seq_len: int = 512
    calib_dataset: str | None = None
    calib_file: str | None = None

    # recuperação
    logit_batches: int = 256
    top_k: int = 128
    epochs: int = 2
    lr: float = 3e-5
    alpha: float = 0.9
    temperature: float = 2.0
    grad_accum: int = 4
    no_checkpointing: bool = False

    # export
    platform: str = "macOS"
    compression: str = "4bit"
    compute_precision: str = "float16"
    max_context_length: int | None = None
    export_dry_run: bool = False

    # controle
    effort: str = "medium"
    device: str | None = None
    measure: bool = False
    skip_recover: bool = False
    skip_export: bool = False
    restart: bool = False
    allow_oversized: bool = False

    @property
    def teacher_dir(self) -> Path:
        return self.out_dir / "teacher"

    @property
    def pruned_dir(self) -> Path:
        return self.out_dir / "pruned"

    @property
    def logits_dir(self) -> Path:
        return self.out_dir / "logits"

    @property
    def student_dir(self) -> Path:
        return self.out_dir / "student"

    @property
    def bundle_dir(self) -> Path:
        return self.out_dir / "bundle"


@dataclass
class Context:
    """Contexto de execução compartilhado entre as etapas do pipeline."""

    opts: RunOptions
    state: RunState
    report: Reporter = print
    events: EventLog = field(default_factory=null_log)
    animate: bool = True
    probe: ModelProbe | None = None
    plan: PrunePlan | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def say(self, msg: str = "") -> None:
        self.report(msg)


# -------------------------------------------------------------- indicadores

# Intervalo mínimo entre eventos de progresso no NDJSON. Sem coalescer, uma
# etapa de milhares de passos afogaria o consumidor em eventos idênticos.
EVENT_INTERVAL = 0.1

INDENT = 13


class _Activity:
    """Indicador de atividade para trecho sem progresso mensurável.

    Anima só quando a execução é interativa: sob `--json` a mesma linha viraria
    lixo no meio do NDJSON, então a mensagem é reportada uma vez e pronto.
    """

    def __init__(self, ctx: Context, text: str, *, indent: int = INDENT) -> None:
        from . import ui

        self._ctx = ctx
        self._text = text
        self._spinner = ui.Spinner(text, indent=indent) if ctx.animate else None

    def __enter__(self) -> _Activity:
        if self._spinner is not None:
            self._spinner.__enter__()
        else:
            self._ctx.say(f"{' ' * INDENT}{self._text}")
        return self

    def update(self, text: str) -> None:
        self._text = text
        if self._spinner is not None:
            self._spinner.update(text)

    def __exit__(self, exc_type: object, *_: object) -> None:
        if self._spinner is not None:
            self._spinner.__exit__(exc_type)


class _Progress:
    """Barra de progresso ligada à etapa, que também alimenta o log de eventos."""

    def __init__(self, ctx: Context, total: int, *, label: str, sid: str,
                 indent: int = INDENT) -> None:
        from . import ui

        self._ctx = ctx
        self._sid = sid
        self._total = max(1, int(total))
        self._bar = ui.Progress(self._total, label=label, indent=indent) if ctx.animate else None
        self._last_event = 0.0
        self._current = 0

    def update(self, current: int, *, suffix: str = "") -> None:
        self._current = current
        if self._bar is not None:
            self._bar.update(current, suffix=suffix)
        agora = time.monotonic()
        if agora - self._last_event >= EVENT_INTERVAL or current >= self._total:
            self._last_event = agora
            self._ctx.events.progress(self._sid, current, self._total)

    def advance(self, n: int = 1, *, suffix: str = "") -> None:
        self.update(self._current + n, suffix=suffix)

    def done(self, *, suffix: str = "") -> None:
        if self._bar is not None:
            self._bar.done(suffix=suffix)


# ----------------------------------------------------------------- guardas


def _resume(ctx: Context, name: str, *, require: tuple[str, ...] = ("dir",)) -> bool:
    """Decide se a etapa concluída pode ser reaproveitada, explicando a decisão."""
    fp = fingerprint(ctx.opts, name)
    gravado = ctx.state.fingerprint_of(name)
    if ctx.state.is_done(name, require=require, fingerprint=fp):
        if ctx.state.stage(name).done and gravado is None:
            ctx.say(f"             {name}: concluída por uma versão anterior, "
                    "sem registro dos parâmetros — reaproveitando")
        return True
    if ctx.state.stage(name).done and gravado not in (None, fp):
        ctx.say(f"             {name}: os parâmetros mudaram desde a última execução "
                "— refazendo a etapa")
    return False


def _disk_free(path: Path) -> int:
    """Espaço livre no volume que contém o caminho, mesmo que ele ainda não exista."""
    p = Path(path).expanduser().absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def _ensure_disk(ctx: Context, needed: int, what: str) -> None:
    """Interrompe antes de começar a escrever quando o volume não comporta a etapa."""
    if needed <= 0:
        return
    free = _disk_free(ctx.opts.out_dir)
    if free >= needed * DISK_MARGIN:
        return
    raise InsufficientResources(
        f"espaço insuficiente para {what}: {needed/GB:.1f} GB estimados mais "
        f"{(DISK_MARGIN - 1) * 100:.0f}% de margem, contra {free/GB:.1f} GB livres "
        f"em {ctx.opts.out_dir}",
        hint="Libere espaço ou aponte -o para um volume com mais capacidade. "
             "Encher o volume do sistema degrada o macOS inteiro, não só esta execução.",
    )


def _training_dtype_bytes(opts: RunOptions) -> int:
    """Bytes por peso no treino: 2 em MPS (float16), 4 nos demais (float32)."""
    try:
        from .loading import pick_device
        device = pick_device(opts.device)
    except Exception:  # noqa: BLE001 — sem torch o plano ainda precisa de um número
        device = opts.device or "mps"
    return 2 if device == "mps" else 4


# --------------------------------------------------------------------- plano


def make_plan(ctx: Context) -> tuple[ModelProbe, PrunePlan | None]:
    """Inspeciona o modelo e calcula o dimensionamento do plano."""
    opts = ctx.opts
    p = probe(opts.model)
    ctx.probe = p

    dtype_bytes = _training_dtype_bytes(opts)
    budget = Budget.for_machine(Machine.detect(opts.out_dir))
    teto = budget.max_params_for_training(dtype_bytes=dtype_bytes)

    target = opts.target_params
    if target is None:
        target = teto
        ctx.say(f"alvo         sugerido pela RAM: {target/1e9:.2f} B parâmetros")
    elif target > teto and not opts.skip_recover:
        aviso = (f"o alvo de {target/1e9:.2f} B excede o teto de treino desta máquina "
                 f"({teto/1e9:.2f} B para {budget.ram_bytes/GB:.0f} GB de orçamento)")
        if not opts.allow_oversized:
            raise InsufficientResources(
                aviso,
                hint=f"Use --target-params {teto:.0f} ou menos, --skip-recover para pular o "
                     "treino, ou --allow-oversized para prosseguir assumindo o risco de "
                     "esgotar a memória na etapa de recuperação.",
            )
        ctx.say(f"aviso        {aviso}")

    if dtype_bytes == 4 and not opts.skip_recover:
        ctx.say("aviso        sem MPS o treino usa float32: o consumo de memória dobra "
                "em relação ao cálculo padrão")

    stages_info = [
        {"id": "fetch", "title": "Download", "rationale": "Baixa os pesos originais do modelo e tokenizador via aria2c."},
        {"id": "prune", "title": "Poda estruturada", "rationale": "Corta camadas e canais do modelo até atingir o tamanho-alvo dimensionado pela RAM."},
        {"id": "logits", "title": "Geração de logits", "rationale": "Gera saídas de calibração do modelo professor para orientar o processo de destilação."},
        {"id": "recover", "title": "Recuperação", "rationale": "Treino de destilação para recuperar a perplexidade perdida na poda estruturada."},
        {"id": "export", "title": "Conversão Core AI", "rationale": "Converte e quantiza o modelo para execução acelerada via Apple Core AI no Neural Engine / GPU."},
    ]
    ctx.events.plan(stages_info)

    total = count_params(p.arch).total
    if total <= target:
        ctx.say(f"plano        modelo atende ao alvo de {target/1e9:.2f} B — sem poda necessária")
        ctx.plan = None
        return p, None

    plan = plan_for_target(p.arch, target)
    ctx.plan = plan
    ctx.say(f"plano        {total/1e9:.2f} B → {plan.target_params/1e9:.2f} B "
            f"({plan.ratio:.2f}× menor)")
    for name, frm, to in plan.changes():
        ctx.say(f"             {name:<22} {frm:>8,} → {to:>8,}")
    return p, plan


# --------------------------------------------------------------------- etapas


def stage_fetch(ctx: Context) -> Path:
    """Etapa 1: Download dos arquivos do modelo."""
    from .fetch import fetch, plan_fetch, require_aria2

    opts, state = ctx.opts, ctx.state
    dest = opts.teacher_dir

    if Path(opts.model).expanduser().is_dir():
        local = Path(opts.model).expanduser().resolve()
        ctx.say(f"fetch        diretório local: {local}")
        state.skip("fetch", "modelo já é local")
        state.stage("fetch").outputs["dir"] = str(local)
        state.save()
        return local

    if state.is_done("fetch", require=("dir",)):
        ctx.say(f"fetch        já concluído: {dest}")
        return dest

    require_aria2()
    fp = plan_fetch(opts.model, dest)
    pending = fp.pending_bytes
    _ensure_disk(ctx, pending, "o download do modelo")
    if pending == 0:
        ctx.say(f"fetch        presente em disco: {fp.total_bytes/GB:.2f} GB")
    else:
        ctx.say(f"fetch        {len(fp.missing())} arquivo(s) pendente(s), {pending/GB:.2f} GB "
                f"de {fp.total_bytes/GB:.2f} GB")

    state.begin("fetch")
    ctx.events.stage_start("fetch", 1, len(STAGES),
                           rationale="Baixa os pesos originais do modelo e tokenizador via aria2c.")
    t0 = time.perf_counter()
    barra = _Progress(ctx, 100, label="baixando", sid="fetch") if pending else None
    try:
        fetch(fp, connections=opts.connections, concurrent=opts.concurrent,
              on_line=_ProgressFilter(ctx.report, bar=barra))
        if barra:
            barra.done(suffix=f"{fp.total_bytes/GB:.2f} GB")
    except Exception as e:
        state.fail("fetch", str(e))
        raise
    dt = time.perf_counter() - t0
    ctx.say(f"             {fp.total_bytes/GB:.2f} GB em {dt:.0f}s")
    state.finish("fetch", outputs={"dir": dest}, metrics={"bytes": fp.total_bytes})
    ctx.events.stage_end("fetch", True, int(dt * 1000))
    return dest


def stage_prune(ctx: Context, teacher_dir: Path) -> Path:
    """Etapa 2: Avaliação de importância e poda estruturada."""
    from .calibration import load_texts, load_texts_from_file, make_batches
    from .loading import load_causal_lm, pick_device, save_pruned
    from .prune import prune_model, score_model

    opts, state = ctx.opts, ctx.state
    out = opts.pruned_dir

    if ctx.plan is None:
        ctx.say("poda         não necessária")
        state.skip("prune", "modelo já cabe no alvo")
        state.stage("prune").outputs["dir"] = str(teacher_dir)
        state.save()
        return teacher_dir

    if _resume(ctx, "prune"):
        ctx.say(f"poda         já concluída: {out}")
        return out

    state.begin("prune")
    ctx.events.stage_start("prune", 2, len(STAGES),
                           rationale="Corta camadas e canais do próprio modelo até o tamanho-alvo.")
    t0 = time.perf_counter()

    device = pick_device(opts.device)
    with _Activity(ctx, f"carregando o modelo (device={device})"):
        model, tokenizer = load_causal_lm(str(teacher_dir), device=device)

    n_texts = opts.calib_batches * opts.batch_size + 8
    texts = (load_texts_from_file(opts.calib_file, limit=n_texts) if opts.calib_file
             else load_texts(dataset=opts.calib_dataset, limit=n_texts))
    batches = make_batches(tokenizer, texts, batch_size=opts.batch_size,
                           seq_len=opts.seq_len, device=device)

    try:
        with _Activity(ctx, f"medindo importância em {opts.calib_batches} lotes"):
            scores = score_model(model, batches, max_batches=opts.calib_batches)
        t = ctx.plan.target
        keep_layers = scores.keep_layers(t.num_hidden_layers)
        report = prune_model(
            model, ctx.plan,
            keep_ffn=scores.top_ffn(t.intermediate_size),
            keep_groups=scores.top_kv_groups(t.num_key_value_heads),
            keep_layers=keep_layers,
        )
        ctx.say(f"             {report.params_before/1e9:.2f} B → "
                f"{report.params_after/1e9:.2f} B ({report.ratio:.2f}× menor)")

        import torch
        with torch.no_grad():
            ids = torch.randint(0, model.config.vocab_size, (1, 16), device=device)
            logits = model(input_ids=ids).logits
        if not torch.isfinite(logits).all():
            raise AguardenteError(
                "o modelo podado produz NaN ou Inf",
                hint="Tente um alvo menos agressivo com --target-params.",
            )
        ctx.say(f"             validação forward ok: {tuple(logits.shape)}")
        save_pruned(model, tokenizer, out)
    except Exception as e:
        state.fail("prune", str(e))
        raise
    finally:
        del model
        _free(device)

    dt = time.perf_counter() - t0
    state.finish("prune", outputs={"dir": out, FINGERPRINT: fingerprint(opts, "prune")},
                 metrics={"params": float(report.params_after), "seconds": dt})
    ctx.metrics["pruned_params"] = float(report.params_after)
    ctx.events.stage_end("prune", True, int(dt * 1000))
    return out


def stage_logits(ctx: Context, teacher_dir: Path) -> Path | None:
    """Etapa 3: Pré-computação dos top-k logits do modelo original."""
    from .calibration import load_texts, load_texts_from_file, make_batches
    from .distill import precompute_logits
    from .distill.teacher import estimate_logit_bytes
    from .loading import load_causal_lm, pick_device
    from .verify import perplexity_on_wikitext

    opts, state = ctx.opts, ctx.state
    out = opts.logits_dir

    if opts.skip_recover:
        state.skip("logits", "recuperação desativada")
        return None

    if _resume(ctx, "logits"):
        ctx.say(f"logits       já concluídos: {out}")
        return out

    n = opts.logit_batches * opts.batch_size
    size, _ = estimate_logit_bytes(n, opts.seq_len, top_k=opts.top_k)
    ctx.say(f"logits       top-{opts.top_k} · {opts.logit_batches} lotes "
            f"≈ {size/GB:.2f} GB em disco")
    _ensure_disk(ctx, size, "a pré-computação de logits")

    state.begin("logits")
    ctx.events.stage_start("logits", 3, len(STAGES),
                           rationale="Gera saídas de calibração do modelo professor para orientar o processo de destilação.")
    t0 = time.perf_counter()
    device = pick_device(opts.device)
    teacher, tokenizer = load_causal_lm(str(teacher_dir), device=device)

    try:
        if opts.measure:
            ppl = perplexity_on_wikitext(teacher, tokenizer, device=device)
            ctx.metrics["ppl_teacher"] = ppl.value
            ctx.say(f"             perplexidade do teacher: {ppl.value:.2f}")

        n_texts = n + 16
        texts = (load_texts_from_file(opts.calib_file, limit=n_texts) if opts.calib_file
                 else load_texts(dataset=opts.calib_dataset, limit=n_texts))
        batches = make_batches(tokenizer, texts, batch_size=opts.batch_size,
                               seq_len=opts.seq_len, device=device)
        barra = _Progress(ctx, opts.logit_batches, label="pré-computando", sid="logits")
        result = precompute_logits(
            teacher, batches, out, top_k=opts.top_k,
            on_progress=lambda k: barra.update(k, suffix=f"shard {k}"),
        )
        barra.done(suffix=f"{result.shards} shards")
    except Exception as e:
        state.fail("logits", str(e))
        raise
    finally:
        del teacher
        _free(device)

    dt = time.perf_counter() - t0
    state.finish("logits", outputs={"dir": out, FINGERPRINT: fingerprint(opts, "logits")},
                 metrics={"seconds": dt})
    ctx.events.stage_end("logits", True, int(dt * 1000))
    return out


def stage_recover(ctx: Context, pruned_dir: Path, logits_dir: Path | None) -> Path:
    """Etapa 4: Treinamento de destilação para recuperação de qualidade."""
    from .distill import RecoveryConfig, recover
    from .distill.teacher import TeacherLogits
    from .loading import load_causal_lm, pick_device, save_pruned
    from .verify import perplexity_on_wikitext, recovery_fraction

    opts, state = ctx.opts, ctx.state
    out = opts.student_dir

    if opts.skip_recover or logits_dir is None:
        ctx.say("recuperação  desativada")
        state.skip("recover", "desativada por opção")
        state.stage("recover").outputs["dir"] = str(pruned_dir)
        state.save()
        return pruned_dir

    if _resume(ctx, "recover"):
        ctx.say(f"recuperação  já concluída: {out}")
        return out

    state.begin("recover")
    ctx.events.stage_start("recover", 4, len(STAGES),
                           rationale="Treino de destilação para recuperar a perplexidade perdida na poda estruturada.")
    t0 = time.perf_counter()

    device = pick_device(opts.device)
    student, tokenizer = load_causal_lm(str(pruned_dir), device=device)
    logits = TeacherLogits.load(logits_dir)

    try:
        if opts.measure:
            ppl = perplexity_on_wikitext(student, tokenizer, device=device)
            ctx.metrics["ppl_pruned"] = ppl.value
            ctx.say(f"             perplexidade pós-poda: {ppl.value:.2f}")

        cfg = RecoveryConfig(
            epochs=opts.epochs, learning_rate=opts.lr, alpha=opts.alpha,
            temperature=opts.temperature, grad_accum=opts.grad_accum,
            gradient_checkpointing=not opts.no_checkpointing,
        )
        ctx.say(f"             {cfg.epochs} época(s) · lr {cfg.learning_rate:g} "
                f"· alpha {cfg.alpha} · T {cfg.temperature}")

        evaluate = None
        if opts.measure:
            evaluate = lambda: perplexity_on_wikitext(student, tokenizer, device=device).value

        with _Activity(ctx, "treinando") as atividade:
            def _passo(s: int, l: float) -> None:
                if s and s % 5 == 0:
                    atividade.update(f"treinando · passo {s} · perda {l:.4f}")
                    ctx.events.progress("recover", s)

            res = recover(student, logits, cfg, device=device, evaluate=evaluate,
                          checkpoint_dir=opts.out_dir / "ckpt", on_step=_passo)
            atividade.update(f"{res.steps} passos em {res.seconds:.0f}s "
                             f"(critério de parada: {res.stopped_by})")

        save_pruned(student, tokenizer, out)

        if opts.measure:
            ppl = perplexity_on_wikitext(student, tokenizer, device=device)
            ctx.metrics["ppl_recovered"] = ppl.value
            if "ppl_teacher" in ctx.metrics and "ppl_pruned" in ctx.metrics:
                frac = recovery_fraction(ctx.metrics["ppl_teacher"],
                                         ctx.metrics["ppl_pruned"], ppl.value)
                ctx.metrics["recovered_fraction"] = frac
                ctx.say(f"             perplexidade recuperada: {ppl.value:.2f} "
                        f"({frac:.1%} da queda)")
    except Exception as e:
        state.fail("recover", str(e))
        raise
    finally:
        del student
        _free(device)

    dt = time.perf_counter() - t0
    state.finish("recover",
                 outputs={"dir": out, FINGERPRINT: fingerprint(opts, "recover")},
                 metrics={k: v for k, v in ctx.metrics.items() if k.startswith("ppl")})
    ctx.events.stage_end("recover", True, int(dt * 1000))
    return out


def stage_export(ctx: Context, model_dir: Path) -> Path | None:
    """Etapa 5: Exportação para o formato .aimodel."""
    from .export import build_command, find_bundle, run_export

    opts, state = ctx.opts, ctx.state
    out = opts.bundle_dir

    if opts.skip_export:
        ctx.say("export       desativado")
        state.skip("export", "desativado por opção")
        return None

    if _resume(ctx, "export"):
        ctx.say(f"export       já concluído: {out}")
        return out

    cmd = build_command(
        model_dir, out, platform=opts.platform, compression=opts.compression,
        compute_precision=opts.compute_precision,
        max_context_length=opts.max_context_length,
        dry_run=opts.export_dry_run,
    )
    ctx.say(f"export       {' '.join(cmd)}")

    state.begin("export")
    ctx.events.stage_start("export", 5, len(STAGES),
                           rationale="Converte e quantiza o modelo para execução acelerada via Apple Core AI no Neural Engine / GPU.")
    t0 = time.perf_counter()
    code = run_export(cmd, on_line=lambda line: ctx.say(f"             {line}"))
    dt = time.perf_counter() - t0

    if code != 0:
        state.fail("export", f"exportador saiu com código {code}")
        raise AguardenteError(
            f"coreai.llm.export falhou (código {code})",
            hint="Rode com --export-dry-run para validar a configuração sem converter.",
        )

    if opts.export_dry_run:
        state.finish("export", outputs={"dry_run": "true"})
        return None

    result = find_bundle(out)
    ctx.say(f"             {result.bundle_dir}")
    if result.aimodel:
        ctx.say(f"             {result.aimodel.name} ({result.size_bytes/GB:.2f} GB)")
        ctx.metrics["bundle_bytes"] = float(result.size_bytes)
    state.finish("export",
                 outputs={"dir": out, "aimodel": result.aimodel or "",
                          FINGERPRINT: fingerprint(opts, "export")},
                 metrics={"seconds": dt, "bytes": float(result.size_bytes)})
    ctx.events.stage_end("export", True, int(dt * 1000))
    return out


def _free(device: str) -> None:
    import gc

    gc.collect()
    try:
        import torch
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- run


def run_pipeline(opts: RunOptions, *, report: Reporter = print,
                 events: EventLog | None = None, animate: bool = True) -> Context:
    """Executa o pipeline completo com suporte a retomada."""
    state = RunState.load_or_create(opts.out_dir, model=opts.model,
                                    target_params=opts.target_params,
                                    restart=opts.restart)
    ctx = Context(opts=opts, state=state, report=report,
                  events=events or null_log(), animate=animate)

    # O lock impede que duas execuções sobre o mesmo -o intercalem etapas e
    # sobrescrevam o estado uma da outra.
    with run_lock(opts.out_dir):
        make_plan(ctx)

        if not opts.skip_export:
            from .export import available
            if not available():
                ctx.say("aviso        coreai.llm.export não está disponível no ambiente")
                ctx.say("             instale via: uv pip install -e '.[pipeline]'")
                ctx.say("             ou use --skip-export para desativar a exportação")
                ctx.say()

        teacher = stage_fetch(ctx)
        pruned = stage_prune(ctx, teacher)
        logits = stage_logits(ctx, teacher)
        student = stage_recover(ctx, pruned, logits)
        stage_export(ctx, student)
    return ctx

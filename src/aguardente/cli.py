"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .arch import count_params
from .budget import (BPW_FP16, BPW_INT4_EMBED_FP16, GB, Budget, Machine,
                     kv_cache_bytes, training_bytes, weights_bytes)
from .errors import AguardenteError
from .plan import plan_for_target
from .pipeline import RunOptions, run_pipeline
from .preflight import Status, blocking, run_all
from .probe import probe
from . import ui
from .ui import StageState

_fmt_params = ui.format_params
_fmt_bytes = ui.format_bytes
_fmt_seconds = ui.format_seconds

_STATE_MAP = {
    "pending": StageState.PENDING,
    "running": StageState.RUNNING,
    "ok": StageState.OK,
    "failed": StageState.FAILED,
    "skipped": StageState.SKIPPED,
}


# ------------------------------------------------------------------ doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    ui.title("Verificação de ambiente", "Diagnóstico dos requisitos do sistema")
    ui.blank()
    ui.explain(
        "Verifica os pré-requisitos necessários para a execução do pipeline.",
        indent=0,
    )
    ui.blank()

    results = run_all()
    width = max(len(r.name) for r in results)
    state_of = {Status.OK: StageState.OK, Status.WARN: StageState.SKIPPED,
                Status.FAIL: StageState.FAILED}
    for r in results:
        ui.check(r.name, state_of[r.status], r.detail, hint=r.hint, width=width)

    fails = blocking(results)
    ui.blank()
    passed = len(results) - len(fails)
    if fails:
        ui.error(f"{len(fails)} de {len(results)} verificações falharam",
                 cause="Existem dependências ou requisitos do sistema pendentes.",
                 action="Siga a instrução ao lado de cada falha e execute "
                        "`aguardente doctor` novamente.")
        return 1
    ui.done(f"{passed} de {len(results)} verificações passaram",
            hint="Execute `aguardente plan <modelo>` para inspecionar um modelo.")
    return 0


# -------------------------------------------------------------------- plan


def cmd_plan(args: argparse.Namespace) -> int:
    """Calcula o plano de poda e dimensionamento do modelo."""
    p = probe(args.model)
    a = p.arch
    b = count_params(a)

    ui.title(f"Plano para {p.name}",
             "Estimativas calculadas a partir dos metadados do modelo")

    ui.header("Modelo")
    ui.field("arquitetura", p.model_type or "desconhecida",
             note=f"{len(p.weight_files)} arquivo(s) de peso")
    ui.field("parâmetros", _fmt_params(p.stored_params), note=f"{p.stored_params:,}")
    ui.field("tamanho original", _fmt_bytes(weights_bytes(p.stored_params, BPW_FP16)))
    if p.gated:
        ui.blank()
        ui.warn("modelo de acesso restrito",
                "Aceite a licença na página do modelo e execute `hf auth login`.")

    err = p.count_error()
    if abs(err) > 0.01:
        ui.warn(f"a contagem diverge {err:+.2%} do total publicado",
                "A arquitetura difere do padrão esperado; o plano pode não ser exato.")

    ui.header("Distribuição de parâmetros")
    ui.table(
        ["componente", "parâmetros", "fatia"],
        [(nome, _fmt_params(getattr(b, nome)), f"{fatia:.1%}")
         for nome, fatia in sorted(b.shares().items(), key=lambda kv: -kv[1])
         if fatia >= 0.001],
        align_right=(1, 2),
    )

    ui.header("Dimensões")
    ui.table(
        ["campo", "valor"],
        [(campo, f"{getattr(a, campo):,}") for campo in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")],
        align_right=(1,),
    )

    m = Machine.detect()
    budget = Budget.for_machine(m)
    ui.header("Ambiente local")
    ui.field("RAM total", f"{m.ram_bytes / GB:.0f} GB",
             note=f"orçamento de {m.usable_ram_bytes / GB:.0f} GB")
    ui.field("disco livre", f"{m.free_disk_bytes / GB:.0f} GB")
    ui.field("treinável até", _fmt_params(budget.max_params_for_training()))
    ui.field("comprimido cabe até", _fmt_params(budget.max_params_for_inference()))

    target = args.target_params or budget.max_params_for_training()
    plan = None
    ui.header("Plano de poda")
    if count_params(a).total <= target:
        ui.explain(
            f"O modelo já atende ao limite de {_fmt_params(target)} parâmetros. "
            "Nenhuma poda estruturada é necessária.",
        )
    else:
        plan = plan_for_target(a, target)
        ui.field("alvo solicitado", _fmt_params(target))
        ui.field("alvo calculado", _fmt_params(plan.target_params),
                 note=f"{plan.ratio:.2f}× menor")
        ui.blank()
        ui.table(
            ["dimensão", "de", "para"],
            [(nome, f"{de:,}", f"{para:,}") for nome, de, para in plan.changes()],
            align_right=(1, 2),
        )

    final = plan.target if plan else a
    final_params = plan.target_params if plan else count_params(a).total

    ui.header("Estimativa de recursos")
    ui.table(
        ["formato", "tamanho"],
        [("original (16 bits por peso)", _fmt_bytes(weights_bytes(final_params, BPW_FP16))),
         ("comprimido (~4,5 bits)", _fmt_bytes(weights_bytes(final_params, BPW_INT4_EMBED_FP16))),
         ("memória por 2.048 tokens", _fmt_bytes(kv_cache_bytes(final, 2048))),
         ("memória por 8.192 tokens", _fmt_bytes(kv_cache_bytes(final, 8192))),
         ("RAM estimada no treino", _fmt_bytes(training_bytes(final_params)))],
        align_right=(1,),
    )

    if plan and training_bytes(final_params) > budget.ram_bytes:
        ui.warn("o treino pode exceder a memória disponível",
                "Considere usar --grad-accum 8 e --seq-len 256, ou aumentar o alvo de parâmetros.")

    ui.blank()
    ui.rule()
    if plan:
        ui.command(f"aguardente run {args.model} -o run/ "
                   f"--target-params {target:.0f} --measure",
                   label="para executar este plano")
    else:
        ui.command(f"aguardente run {args.model} -o run/ --measure",
                   label="para converter este modelo")
    return 0


# ------------------------------------------------------------------- fetch


def cmd_fetch(args: argparse.Namespace) -> int:
    """Executa o download dos arquivos do modelo."""
    from .fetch import fetch, plan_fetch, require_aria2

    require_aria2()
    fp = plan_fetch(args.model, args.out)
    pendentes = fp.missing()

    ui.title(f"Download de {args.model}")
    ui.explain(
        "Download dos arquivos do modelo via aria2c com suporte a retomada.",
        indent=0,
    )
    ui.blank()
    ui.field("arquivos no repositório", str(len(fp.files)))
    ui.field("tamanho total", _fmt_bytes(fp.total_bytes))

    if not pendentes:
        ui.done("arquivos já estão presentes no disco", hint=str(fp.dest))
        return 0

    ui.field("pendente", f"{len(pendentes)} arquivo(s)",
             note=_fmt_bytes(fp.pending_bytes))
    ui.blank()

    t0 = time.perf_counter()
    fetch(fp, connections=args.connections, concurrent=args.concurrent,
          on_line=lambda line: ui.step(line))
    ui.done(f"concluído em {_fmt_seconds(time.perf_counter() - t0)}",
            hint=str(fp.dest))
    return 0


# --------------------------------------------------------------------- run


def _options_from(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        model=args.model,
        out_dir=Path(args.out).expanduser(),
        target_params=args.target_params,
        connections=args.connections,
        concurrent=args.concurrent,
        calib_batches=args.calib_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        calib_dataset=args.calib_dataset,
        calib_file=args.calib_file,
        logit_batches=args.logit_batches,
        top_k=args.top_k,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        temperature=args.temperature,
        grad_accum=args.grad_accum,
        no_checkpointing=args.no_checkpointing,
        platform=args.platform,
        compression=args.compression,
        compute_precision=args.compute_precision,
        max_context_length=args.max_context_length,
        export_dry_run=args.export_dry_run,
        device=args.device,
        measure=args.measure,
        skip_recover=args.skip_recover,
        skip_export=args.skip_export,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Executa o pipeline completo."""
    opts = _options_from(args)

    ui.title(f"aguardente · {opts.model}",
             "Poda estruturada, destilação e conversão para Core AI")

    if not args.skip_checks:
        results = run_all()
        fails = blocking(results)
        if fails:
            ui.error(
                "o ambiente não atende aos requisitos necessários",
                cause="; ".join(f"{r.name}: {r.detail}" for r in fails),
                action="Execute `aguardente doctor` para ver instruções de correção, "
                       "ou use --skip-checks para prosseguir sem validação.",
            )
            return 1

    ui.blank()
    ui.explain(
        "O pipeline é executado em etapas com persistência de estado em disco.",
        indent=0,
    )
    ui.blank()
    ui.field("destino", str(opts.out_dir))
    if opts.target_params:
        ui.field("alvo", _fmt_params(opts.target_params))
    ui.rule()

    inicio = time.perf_counter()
    ctx = run_pipeline(opts)
    total = time.perf_counter() - inicio

    ui.blank()
    ui.rule()
    ui.header("Resumo")
    etapas = ctx.state.summary()
    for n, (nome, status, segundos) in enumerate(etapas, start=1):
        ui.stage(n, len(etapas), nome,
                 _STATE_MAP.get(status.value, StageState.PENDING),
                 seconds=segundos)
    ui.blank()
    ui.field("tempo total", _fmt_seconds(total))

    mostrar = {k: v for k, v in ctx.metrics.items()
               if k.startswith("ppl") or k in ("recovered_fraction", "bundle_bytes")}
    if mostrar:
        ui.header("Métricas de qualidade")
        linhas = []
        rotulos = {"ppl_teacher": "modelo original", "ppl_pruned": "após poda",
                   "ppl_recovered": "após recuperação"}
        for chave, rotulo in rotulos.items():
            if chave in ctx.metrics:
                linhas.append((rotulo, f"{ctx.metrics[chave]:.2f}"))
        if linhas:
            ui.table(["etapa", "perplexidade"], linhas, align_right=(1,))
        if "recovered_fraction" in ctx.metrics:
            ui.blank()
            ui.field("queda recuperada", f"{ctx.metrics['recovered_fraction']:.1%}")
        if "bundle_bytes" in ctx.metrics:
            ui.field("tamanho final", _fmt_bytes(ctx.metrics["bundle_bytes"]))

    bundle = ctx.state.output("export", "dir")
    ui.blank()
    ui.rule()
    if bundle:
        ui.done("pipeline concluído", hint=str(bundle))
        ui.blank()
        ui.command(f'swift run -c release llm-runner --model {bundle} --prompt "Olá"',
                   label="para testar o modelo gerado")
    else:
        ui.done(f"pipeline concluído em {_fmt_seconds(total)}",
                hint=str(opts.out_dir))
    return 0


# ------------------------------------------------------------------ status


def cmd_status(args: argparse.Namespace) -> int:
    """Exibe o estado de uma execução existente."""
    from .state import RunState

    d = Path(args.out).expanduser()
    if not (d / "state.json").is_file():
        ui.error(f"nenhuma execução encontrada em {d}",
                 cause="O arquivo state.json não existe neste diretório.",
                 action="Verifique o caminho ou inicie uma execução com "
                        "`aguardente run <modelo> -o <diretório>`.")
        return 1

    st = RunState.load_or_create(d)
    ui.title("Estado da execução", str(d))
    ui.blank()
    ui.field("modelo", st.model or "—")
    if st.target_params:
        ui.field("alvo", _fmt_params(st.target_params))

    ui.header("Etapas")
    total = len(st.summary()) or 1
    for n, (nome, status, segundos) in enumerate(st.summary(), start=1):
        estado = _STATE_MAP.get(status.value, StageState.PENDING)
        ui.stage(n, total, nome, estado, seconds=segundos,
                 detail=st.stage(nome).error)

    pendentes = [n for n, s, _ in st.summary() if s.value not in ("ok", "skipped")]
    ui.blank()
    if pendentes:
        ui.note(f"etapas pendentes: {', '.join(pendentes)}")
        ui.command(f"aguardente run {st.model} -o {d}", label="para retomar")
    else:
        ui.done("todas as etapas foram concluídas")
    return 0


# ----------------------------------------------------------------- parser


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    """Argumentos compartilhados do pipeline."""
    p.add_argument("--target-params", type=lambda s: int(float(s)),
                   help="alvo de parâmetros (ex.: 1.4e9). Padrão: baseado na RAM disponível")
    p.add_argument("--connections", type=int, default=8,
                   help="número de conexões por servidor no aria2c")
    p.add_argument("--concurrent", type=int, default=4,
                   help="número de downloads simultâneos no aria2c")
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--calib-dataset", help="dataset de calibração (namespace/name)")
    p.add_argument("--calib-file", help="arquivo .txt local com amostras separadas por linha em branco")
    p.add_argument("--logit-batches", type=int, default=256)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--platform", default="macOS",
                   choices=["macOS", "iOS", "watchOS", "visionOS", "tvOS"])
    p.add_argument("--compression", default="4bit")
    p.add_argument("--compute-precision", default="float16")
    p.add_argument("--max-context-length", type=int)
    p.add_argument("--export-dry-run", action="store_true",
                   help="valida argumentos de exportação sem converter o modelo")
    p.add_argument("--device", help="dispositivo de execução (mps, cpu)")
    p.add_argument("--measure", action="store_true",
                   help="avalia a perplexidade durante as etapas do pipeline")
    p.add_argument("--skip-recover", action="store_true")
    p.add_argument("--skip-export", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aguardente",
        description="Poda estruturada, destilação e conversão para Core AI.",
    )
    p.add_argument("--version", action="version", version=f"aguardente {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="verifica o ambiente e dependências")
    d.set_defaults(func=cmd_doctor)

    pl = sub.add_parser("plan", help="inspeciona o modelo e calcula o plano de poda sem baixar pesos")
    pl.add_argument("model", help="identificador do Hugging Face ou diretório local")
    pl.add_argument("--target-params", type=lambda s: int(float(s)))
    pl.set_defaults(func=cmd_plan)

    ft = sub.add_parser("fetch", help="baixa os arquivos do modelo via aria2c")
    ft.add_argument("model", help="identificador do Hugging Face")
    ft.add_argument("-o", "--out", required=True, help="diretório de destino")
    ft.add_argument("--connections", type=int, default=8)
    ft.add_argument("--concurrent", type=int, default=4)
    ft.set_defaults(func=cmd_fetch)

    rn = sub.add_parser("run", help="executa o pipeline completo (download, poda, recuperação e conversão)")
    rn.add_argument("model", help="identificador do Hugging Face ou diretório local")
    rn.add_argument("-o", "--out", required=True, help="diretório da execução")
    rn.add_argument("--skip-checks", action="store_true",
                    help="ignora verificações de pré-requisitos de ambiente")
    _add_pipeline_args(rn)
    rn.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="exibe o estado e progresso de uma execução")
    st.add_argument("-o", "--out", required=True, help="diretório da execução")
    st.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AguardenteError as e:
        print(f"\nerro: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nexecução interrompida pelo usuário — estado salvo para retomada", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

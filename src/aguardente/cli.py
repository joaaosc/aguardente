"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .arch import count_params
from .budget import (BPW_FP16, BPW_INT4_EMBED_FP16, GB, Budget, Machine,
                     kv_cache_bytes, training_bytes, weights_bytes)
from . import effort
from .errors import AguardenteError
from .events import stdout_log
from .errors import PlanImpossible
from .plan import plan_for_target, shrink
from .pipeline import DISK_MARGIN, RunOptions, run_pipeline
from .preflight import Status, blocking, run_all
from .probe import probe
from .source import resolve_source
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

# Variável de ambiente que libera o traceback completo nas falhas inesperadas.
DEBUG_ENV = "AGUARDENTE_DEBUG"

# Subdiretórios derivados de uma execução, descartados por --restart.
ARTIFACT_DIRS = ("teacher", "pruned", "logits", "student", "bundle", "ckpt")


# ------------------------------------------------------------------ doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    ui.title("Verificação de ambiente", "Diagnóstico dos requisitos do sistema")
    ui.blank()
    ui.explain(
        "Verifica os pré-requisitos necessários para a execução do pipeline.",
        indent=0,
    )
    ui.blank()

    verbose = getattr(args, "verbose", False)
    results = run_all()
    width = max(len(r.name) for r in results)
    state_of = {Status.OK: StageState.OK, Status.WARN: StageState.SKIPPED,
                Status.FAIL: StageState.FAILED}
    for r in results:
        ui.check(r.name, state_of[r.status], r.detail, hint=r.hint, width=width)
        if verbose and r.debug:
            for linha in r.debug.splitlines():
                ui.step(linha, indent=6 + width)

    fails = blocking(results)
    ui.blank()
    # Avisos não são aprovações: contá-los como tal descrevia como íntegro um
    # ambiente em que `aguardente run` seria bloqueado logo em seguida.
    passed = sum(1 for r in results if r.status is Status.OK)
    avisos = [r for r in results if r.status is Status.WARN]
    if fails:
        acao = ("Siga a instrução ao lado de cada falha e execute "
                "`aguardente doctor` novamente.")
        if not verbose and any(r.debug for r in fails):
            acao += " Use `aguardente doctor --verbose` para ver a saída bruta dos comandos."
        ui.error(f"{len(fails)} de {len(results)} verificações falharam",
                 cause="Existem dependências ou requisitos do sistema pendentes.",
                 action=acao)
        return 1
    resumo = f"{passed} de {len(results)} verificações passaram"
    if avisos:
        resumo += f" · {len(avisos)} aviso(s): {', '.join(r.name for r in avisos)}"
        ui.done(resumo,
                hint="Avisos não impedem `plan`, mas a stack do pipeline é requisito de "
                     "`aguardente run`.")
        return 0
    ui.done(resumo, hint="Execute `aguardente plan <modelo>` para inspecionar um modelo.")
    return 0


# -------------------------------------------------------------------- plan


def cmd_plan(args: argparse.Namespace) -> int:
    """Calcula o plano de poda e dimensionamento do modelo."""
    avisos: list[str] = []
    args.model = resolve_source(args.model, report=avisos.append)
    p = probe(args.model)
    a = p.arch
    b = count_params(a)

    ui.title(f"Plano para {p.name}",
             "Estimativas calculadas a partir dos metadados do modelo")
    for aviso in avisos:
        ui.step(aviso)

    ui.header("Modelo")
    ui.field("arquitetura", p.model_type or "desconhecida",
             note=f"{len(p.weight_files)} arquivo(s) de peso")
    ui.field("parâmetros", _fmt_params(p.stored_params), note=f"{p.stored_params:,}")
    ui.field("tamanho original", _fmt_bytes(weights_bytes(p.stored_params, BPW_FP16)))
    if p.gated:
        ui.blank()
        ui.warn("modelo de acesso restrito",
                "Aceite a licença na página do modelo e execute `hf auth login`.")

    from .textonly import emitted_model_type
    try:
        alvo_export = emitted_model_type(p.text_model_type)
        if alvo_export != p.text_model_type:
            ui.field("exporta como", alvo_export, note=f"de {p.text_model_type}")
    except AguardenteError as e:
        ui.blank()
        ui.warn(e.message, e.hint or "")
        alvo_export = None

    if p.is_multimodal:
        ui.header("Multimodalidade")
        ui.explain(
            "O pipeline poda e destila um decoder causal de texto. A torre de visão "
            "e o projetor são descartados: o modelo resultante não enxerga imagens.",
        )
        ui.blank()
        ui.field("decoder de texto", p.layout.prefix or "raiz",
                 note=f"{p.layout.num_layers} camadas")
        ui.field("mantido", _fmt_params(p.text_params),
                 note=f"{p.text_params / p.stored_params:.1%} do checkpoint")
        ui.field("descartado", _fmt_params(p.dropped_params),
                 note=f"{p.dropped_params / p.stored_params:.1%}")
        if alvo_export:
            ui.field("arquitetura emitida", alvo_export, note=f"de {p.text_model_type}")

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

    if args.other_ram_gb is not None or args.other_disk_gb is not None:
        if args.other_ram_gb is None or args.other_disk_gb is None:
            raise AguardenteError(
                "--other-ram-gb e --other-disk-gb precisam ser usados juntos",
                hint="Informe os dois para avaliar a outra máquina.")
        _secao_outra_maquina(a, p, args.other_ram_gb, args.other_disk_gb,
                             args.target_params)

    teto = budget.max_params_for_training()
    target = args.target_params or teto
    if args.target_params and args.target_params > teto:
        ui.blank()
        ui.warn(f"o alvo de {_fmt_params(args.target_params)} excede o teto de treino "
                f"desta máquina ({_fmt_params(teto)})",
                "A etapa de recuperação tende a esgotar a memória. Reduza o alvo, use "
                "--skip-recover, ou --allow-oversized para assumir o risco.")
    plan = None
    ui.header("Plano de poda")
    if count_params(a).total <= target:
        ui.explain(
            f"O modelo já atende ao limite de {_fmt_params(target)} parâmetros. "
            "Nenhuma poda estruturada é necessária.",
        )
    else:
        try:
            plan = plan_for_target(a, target)
        except PlanImpossible as e:
            # O piso de poda é imposto pelas embeddings e pelos limites por eixo.
            # Quando ele excede o que a máquina treina, dizer isso aqui vale mais
            # do que abortar: o modelo ainda pode ser podado e exportado.
            piso = count_params(shrink(a, 1.0)).total
            ui.field("alvo solicitado", _fmt_params(target))
            ui.field("piso de poda", _fmt_params(piso),
                     note="limite das embeddings e dos cortes por eixo")
            ui.blank()
            ui.warn(e.message.split("\n")[0],
                    "A recuperação por destilação não é executável nesta máquina para "
                    "este modelo. Use --skip-recover para podar e exportar sem treinar, "
                    "ou escolha um modelo base menor.")
        else:
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

    nivel = effort.get(getattr(args, "effort", None))
    lote = getattr(args, "batch_size", 2)

    ui.header("Estimativa de recursos")
    ui.table(
        ["formato", "tamanho"],
        [("original (16 bits por peso)", _fmt_bytes(weights_bytes(final_params, BPW_FP16))),
         ("comprimido (~4,5 bits)", _fmt_bytes(weights_bytes(final_params, BPW_INT4_EMBED_FP16))),
         ("memória por 2.048 tokens", _fmt_bytes(kv_cache_bytes(final, 2048))),
         ("memória por 8.192 tokens", _fmt_bytes(kv_cache_bytes(final, 8192))),
         ("RAM estimada no treino", _fmt_bytes(training_bytes(final_params))),
         (f"logits em disco (--effort {nivel.name})", _fmt_bytes(nivel.logit_bytes(lote)))],
        align_right=(1,),
    )

    _effort_panel(nivel, [], batch_size=lote, animate=not getattr(args, "no_anim", False))

    if plan and training_bytes(final_params) > budget.ram_bytes:
        ui.warn("o treino pode exceder a memória disponível",
                "Considere usar --grad-accum 8 e --seq-len 256, ou aumentar o alvo de parâmetros.")

    ui.blank()
    ui.rule()
    if plan:
        # O comando sugerido precisa ser executável: sem a flag, `run` recusaria
        # o alvo que este mesmo relatório acabou de exibir.
        extra = " --allow-oversized" if target > teto else ""
        ui.command(f"aguardente run {args.model} -o run/ "
                   f"--target-params {target:.0f}{extra} --effort {nivel.name} --measure",
                   label="para executar este plano")
    else:
        ui.command(f"aguardente run {args.model} -o run/ --effort {nivel.name} --measure",
                   label="para converter este modelo")
    return 0


def _secao_outra_maquina(a, p, ram_gb: float, disk_gb: float,
                         target_params: int | None) -> None:
    """Avalia conversão e destilação numa máquina descrita só por RAM e disco.

    Não toca a máquina atual nem baixa nada: usa as mesmas contas do plano
    local (`Budget`, `plan_for_target`), só que sobre uma `Machine` hipotética.
    A pergunta que decide viabilidade de destilação não é "cabe o modelo na
    RAM", e sim "o piso de poda desta arquitetura cabe no teto de treino" —
    são números diferentes, e o segundo costuma ser bem menor.
    """
    m = Machine.other(ram_gb=ram_gb, disk_gb=disk_gb)
    budget = Budget.for_machine(m)
    teto = budget.max_params_for_training()
    alvo = target_params or teto

    ui.header(f"Outra máquina ({ram_gb:g} GB RAM, {disk_gb:g} GB disco)")
    ui.field("RAM utilizável", f"{m.usable_ram_bytes / GB:.1f} GB")
    ui.field("teto de treino", _fmt_params(teto))

    plano = None
    if count_params(a).total <= alvo:
        destilacao_viavel = True
        final_params = count_params(a).total
    else:
        try:
            plano = plan_for_target(a, alvo)
            destilacao_viavel = True
            final_params = plano.target_params
        except PlanImpossible:
            destilacao_viavel = False
            final_params = count_params(shrink(a, 1.0)).total
            ui.field("piso de poda", _fmt_params(final_params),
                     note="limite das embeddings e dos cortes por eixo")

    # Estimativa de pico de disco: soma das etapas sem descartar nada no
    # caminho, de propósito — é o teto do que o pipeline pode vir a usar de
    # uma vez, não uma previsão exata do que sobra em cada etapa.
    download = weights_bytes(p.stored_params, BPW_FP16)
    extracao = weights_bytes(p.text_params, BPW_FP16) if p.is_multimodal else 0
    poda = weights_bytes(final_params, BPW_FP16)
    bundle = weights_bytes(final_params, BPW_INT4_EMBED_FP16)
    pico = download + extracao + poda + bundle
    disco_viavel = disk_gb * GB >= pico * DISK_MARGIN

    ui.field("disco no pico (estimado)", _fmt_bytes(pico),
             note=f"download {_fmt_bytes(download)}"
                  + (f" + extração {_fmt_bytes(extracao)}" if extracao else "")
                  + f" + poda {_fmt_bytes(poda)} + bundle {_fmt_bytes(bundle)}")

    ui.blank()
    ui.field("conversão (poda + export)", "viável" if disco_viavel else "inviável",
             note=None if disco_viavel else
             f"faltam ~{_fmt_bytes(pico * DISK_MARGIN - disk_gb * GB)} de disco")
    ui.field("destilação (recuperação)", "viável" if destilacao_viavel else "inviável",
             note=None if destilacao_viavel else
             "o piso de poda excede o teto de treino desta máquina — só "
             "--skip-recover chegaria a um resultado nela")


# ------------------------------------------------------------------ effort


def _effort_panel(nivel, sobrescritos, *, batch_size: int = 2,
                  animate: bool = True) -> None:
    """Painel do nível escolhido: escala, o que muda e o que custa."""
    ui.header("Esforço da destilação")
    ui.meter_line(effort.index(nivel), total=len(effort.LEVELS),
                  label=f"{ui.bold(nivel.name)}  {ui.dim('· ' + nivel.label)}",
                  animate=animate)
    ui.explain(nivel.summary)
    ui.blank()
    ui.field("custo relativo", f"{effort.relative_cost(nivel):.2f}× do nível "
                               f"{effort.DEFAULT}")
    ui.field("logits em disco", _fmt_bytes(nivel.logit_bytes(batch_size)))
    ui.field("épocas de recuperação", str(nivel.epochs))
    ui.field("profundidade top-k", str(nivel.top_k))
    if sobrescritos:
        ui.blank()
        ui.note(f"informado na linha de comando, com precedência sobre o preset: "
                f"{', '.join('--' + c.replace('_', '-') for c in sobrescritos)}")


def cmd_effort(args: argparse.Namespace) -> int:
    """Explica a escala de esforço sem tocar em nenhum modelo."""
    animar = not getattr(args, "no_anim", False)
    lote = getattr(args, "batch_size", 2)

    ui.title("Esforço da destilação",
             "Quanto trabalho investir para recuperar a qualidade perdida na poda")
    ui.blank()
    ui.explain(
        "O nível de esforço é ortogonal ao alvo de poda: --target-params decide o "
        "tamanho do resultado, --effort decide o cuidado com que se chega nele. "
        "Um alvo agressivo com esforço baixo é o caminho mais rápido para um modelo ruim.",
        indent=0,
    )

    ui.header("Escala")
    for nivel in effort.LEVELS:
        ui.blank()
        ui.meter_line(effort.index(nivel), total=len(effort.LEVELS),
                      label=f"{ui.bold(f'{nivel.name:<7}')} {ui.dim('· ' + nivel.label)}",
                      animate=animar)
        ui.explain(nivel.summary, indent=8)
        ui.explain(nivel.when, indent=8)

    ui.header("Comparação")
    ui.table(
        ["nível", "épocas", "top-k", "seq", "lotes", "logits em disco", "custo"],
        [(l["name"], str(l["epochs"]), str(l["top_k"]), str(l["seq_len"]),
          str(l["logit_batches"]), _fmt_bytes(l["logit_bytes"]), f"{l['cost']:.2f}×")
         for l in effort.comparison(lote)],
        align_right=(1, 2, 3, 4, 5, 6),
    )
    ui.blank()
    ui.explain(
        f"Disco estimado com --batch-size {lote}. O custo é o tempo relativo ao nível "
        f"{effort.DEFAULT}, calculado pelos tokens processados em cada etapa.",
    )

    ui.blank()
    ui.rule()
    ui.command("aguardente run <modelo> -o run/ --effort high",
               label="para usar um nível específico")
    return 0


# ------------------------------------------------------------------- fetch


def cmd_fetch(args: argparse.Namespace) -> int:
    """Executa o download dos arquivos do modelo."""
    from .fetch import fetch, plan_fetch, require_aria2

    avisos: list[str] = []
    args.model = resolve_source(args.model, report=avisos.append)

    require_aria2()
    fp = plan_fetch(args.model, args.out)
    pendentes = fp.missing()

    ui.title(f"Download de {args.model}")
    for aviso in avisos:
        ui.step(aviso)
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


# ----------------------------------------------------------------- extract


def cmd_extract(args: argparse.Namespace) -> int:
    """Extrai o decoder causal de texto de um checkpoint multimodal."""
    import json

    from . import textonly
    from .fetch import fetch, plan_fetch, require_aria2

    avisos: list[str] = []
    args.model = resolve_source(args.model, report=avisos.append)
    origem = Path(args.model).expanduser()
    destino = Path(args.out).expanduser()

    if not origem.is_dir():
        require_aria2()
        origem = destino / "teacher"
        fp = plan_fetch(args.model, origem)
        ui.title(f"Extração de {args.model}")
        for aviso in avisos:
            ui.step(aviso)
        ui.field("download", _fmt_bytes(fp.pending_bytes),
                 note=f"de {_fmt_bytes(fp.total_bytes)}")
        if fp.missing():
            fetch(fp, on_line=lambda linha: ui.step(linha))
    else:
        ui.title(f"Extração de {origem.name}")
        for aviso in avisos:
            ui.step(aviso)

    p = probe(str(origem))
    if p.layout is None or not p.layout.needs_extraction:
        ui.done("o decoder já está na raiz canônica — nada a extrair",
                hint=str(origem))
        return 0

    ui.blank()
    ui.field("decoder", p.layout.prefix or "raiz", note=f"{p.layout.num_layers} camadas")
    ui.field("mantido", _fmt_params(p.text_params))
    ui.field("descartado", _fmt_params(p.dropped_params),
             note=f"{p.dropped_params / p.stored_params:.1%} do checkpoint")

    saida = destino / "teacher-text"
    cfg = json.loads((origem / "config.json").read_text())
    emitido = textonly.build_config(cfg, p.arch, p.layout)

    ui.blank()
    ui.field("arquitetura emitida", emitido.emitted_type,
             note=emitido.config["architectures"][0])
    if emitido.defaults_used:
        ui.field("preenchido por default", ", ".join(emitido.defaults_used))
    ui.blank()

    t0 = time.perf_counter()
    relatorio = textonly.extract_weights(origem, saida, p.layout,
                                         reuse_shards=args.discard_source_weights)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "config.json").write_text(
        json.dumps(emitido.config, indent=2, ensure_ascii=False) + "\n")
    textonly.copy_auxiliary(origem, saida)
    textonly.verify_extraction(saida, p.arch)

    ui.field("tensores", f"{relatorio.tensors:,}",
             note=f"{len(relatorio.shards)} shard(s)")
    if relatorio.reused:
        ui.field("shards reaproveitados", str(len(relatorio.reused)),
                 note=f"{_fmt_bytes(relatorio.saved_bytes)} não copiados")
    ui.blank()
    ui.rule()
    ui.done(f"extraído em {_fmt_seconds(time.perf_counter() - t0)}", hint=str(saida))
    ui.blank()
    ui.command(f"aguardente run {saida} -o run/ --measure",
               label="para podar e converter o decoder")
    return 0


# --------------------------------------------------------------------- run


def _options_from(args: argparse.Namespace) -> RunOptions:
    nivel = effort.get(getattr(args, "effort", None))
    informado = {campo: getattr(args, campo, None) for campo in effort.CONTROLLED}
    valores, sobrescritos = effort.resolve(nivel, informado)
    # Guardados no Namespace para que a apresentação saiba o que veio do preset
    # e o que o usuário digitou, sem recalcular a resolução.
    args.effort_level = nivel
    args.effort_overrides = sobrescritos
    return RunOptions(
        model=args.model,
        out_dir=Path(args.out).expanduser(),
        target_params=args.target_params,
        connections=args.connections,
        concurrent=args.concurrent,
        batch_size=args.batch_size,
        calib_dataset=args.calib_dataset,
        calib_file=args.calib_file,
        effort=nivel.name,
        no_checkpointing=args.no_checkpointing,
        **valores,
        platform=args.platform,
        compression=args.compression,
        compute_precision=args.compute_precision,
        max_context_length=args.max_context_length,
        export_dry_run=args.export_dry_run,
        device=args.device,
        measure=args.measure,
        skip_recover=args.skip_recover,
        skip_export=args.skip_export,
        restart=getattr(args, "restart", False),
        allow_oversized=getattr(args, "allow_oversized", False),
    )


def _discard_run_dir(out_dir: Path) -> list[str]:
    """Remove o estado e os artefatos de uma execução anterior, listando o descartado."""
    removidos = []
    for nome in ("state.json", "run.lock"):
        alvo = out_dir / nome
        if alvo.exists():
            alvo.unlink()
            removidos.append(nome)
    for nome in ARTIFACT_DIRS:
        alvo = out_dir / nome
        if alvo.is_dir():
            shutil.rmtree(alvo)
            removidos.append(f"{nome}/")
    return removidos


def cmd_run(args: argparse.Namespace) -> int:
    """Executa o pipeline completo."""
    is_json = getattr(args, "json", False)
    # Em modo --json o stdout é o fluxo de eventos NDJSON: o aviso de
    # resolução, se houver, sai só como linha de comentário do relatório final.
    avisos: list[str] = []
    args.model = resolve_source(args.model,
                                report=(avisos.append if not is_json else (lambda _m: None)))
    opts = _options_from(args)
    event_log = stdout_log() if is_json else None

    if not is_json:
        ui.title(f"aguardente · {opts.model}",
                 "Poda estruturada, destilação e conversão para Core AI")
        for aviso in avisos:
            ui.step(aviso)

    if args.restart and opts.out_dir.is_dir():
        removidos = _discard_run_dir(opts.out_dir)
        if removidos and not is_json:
            ui.blank()
            ui.warn(f"--restart descartou: {', '.join(removidos)}",
                    "Todas as etapas serão refeitas do zero.")

    if not args.skip_checks:
        # A stack do pipeline é informativa no `doctor` e requisito aqui: sem
        # torch a execução avançaria até morrer num import, possivelmente depois
        # do download já ter consumido tempo e banda.
        results = run_all(path=opts.out_dir)
        fails = blocking(results, warn_as_fail=("stack do pipeline",))
        if fails:
            cause_str = "; ".join(f"{r.name}: {r.detail}" for r in fails)
            if is_json:
                event_log.error("preflight", "o ambiente não atende aos requisitos necessários", hint=cause_str)
            else:
                ui.error(
                    "o ambiente não atende aos requisitos necessários",
                    cause=cause_str,
                    action="Execute `aguardente doctor` para ver instruções de correção, "
                           "ou use --skip-checks para prosseguir sem validação.",
                )
            return 1

    if not is_json:
        ui.blank()
        ui.explain(
            "O pipeline é executado em etapas com persistência de estado em disco.",
            indent=0,
        )
        ui.blank()
        ui.field("destino", str(opts.out_dir))
        if opts.target_params:
            ui.field("alvo", _fmt_params(opts.target_params))
        if not opts.skip_recover:
            _effort_panel(args.effort_level, args.effort_overrides,
                          batch_size=opts.batch_size)
        ui.blank()
        ui.rule()

    if is_json:
        nivel = args.effort_level
        event_log.emit("effort", name=nivel.name, label=nivel.label,
                       summary=nivel.summary, cost=effort.relative_cost(nivel),
                       overrides=args.effort_overrides, values=nivel.values())

    inicio = time.perf_counter()
    report_func = (lambda msg="": event_log.log("pipeline", msg) if msg else None) if is_json else print
    ctx = run_pipeline(opts, report=report_func, events=event_log,
                       animate=not is_json)
    total = time.perf_counter() - inicio

    if is_json:
        return 0

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


# Os padrões das opções abaixo são `None` de propósito: só assim se distingue
# "não informado" de "informado com o mesmo valor do preset", e o nível de
# esforço pode preencher o que faltou sem passar por cima da escolha de quem
# digitou a opção. O valor efetivo aparece no painel de esforço.
PRESET = "definido por --effort"


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    """Argumentos compartilhados do pipeline."""
    p.add_argument("--effort", choices=effort.NAMES, default=effort.DEFAULT,
                   help="agressividade da destilação: quanto trabalho investir para "
                        f"recuperar a qualidade perdida na poda (padrão: {effort.DEFAULT})")
    p.add_argument("--target-params", type=lambda s: int(float(s)),
                   help="alvo de parâmetros (ex.: 1.0e9). Padrão: baseado na RAM disponível")
    p.add_argument("--connections", type=int, default=8,
                   help="número de conexões por servidor no aria2c")
    p.add_argument("--concurrent", type=int, default=4,
                   help="número de downloads simultâneos no aria2c")
    p.add_argument("--calib-batches", type=int, help=PRESET)
    p.add_argument("--batch-size", type=int, default=2,
                   help="amostras por lote; restrição de memória, não de esforço (padrão: 2)")
    p.add_argument("--seq-len", type=int, help=PRESET)
    p.add_argument("--calib-dataset", help="dataset de calibração (namespace/name)")
    p.add_argument("--calib-file", help="arquivo .txt local com amostras separadas por linha em branco")
    p.add_argument("--logit-batches", type=int, help=PRESET)
    p.add_argument("--top-k", type=int, help=PRESET)
    p.add_argument("--epochs", type=int, help=PRESET)
    p.add_argument("--lr", type=float, help=PRESET)
    p.add_argument("--alpha", type=float, help=PRESET)
    p.add_argument("--temperature", type=float, help=PRESET)
    p.add_argument("--grad-accum", type=int, help=PRESET)
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
    p.add_argument("--json", action="store_true",
                   help="emite eventos estruturados em formato NDJSON no stdout para a GUI")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aguardente",
        description="Poda estruturada, destilação e conversão para Core AI.",
    )
    p.add_argument("--version", action="version", version=f"aguardente {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="verifica o ambiente e dependências")
    d.add_argument("-v", "--verbose", action="store_true",
                   help="exibe a saída bruta dos comandos que falharam")
    d.set_defaults(func=cmd_doctor)

    pl = sub.add_parser("plan", help="inspeciona o modelo e calcula o plano de poda sem baixar pesos")
    pl.add_argument("model", help="identificador do Hugging Face (namespace/nome), "
                    "URL do Hugging Face ou do GitHub, ou diretório local")
    pl.add_argument("--target-params", type=lambda s: int(float(s)))
    pl.add_argument("--other-ram-gb", type=float,
                    help="RAM de outra máquina, em GB — avalia a viabilidade lá, "
                         "sem tocar na máquina atual. Exige --other-disk-gb junto")
    pl.add_argument("--other-disk-gb", type=float,
                    help="disco livre de outra máquina, em GB. Exige --other-ram-gb junto")
    pl.add_argument("--effort", choices=effort.NAMES, default=effort.DEFAULT,
                    help=f"nível de esforço a dimensionar (padrão: {effort.DEFAULT})")
    pl.add_argument("--batch-size", type=int, default=2)
    pl.add_argument("--no-anim", action="store_true", help="não anima os medidores")
    pl.set_defaults(func=cmd_plan)

    ft = sub.add_parser("fetch", help="baixa os arquivos do modelo via aria2c")
    ft.add_argument("model", help="identificador do Hugging Face, ou URL do Hugging "
                    "Face ou do GitHub")
    ft.add_argument("-o", "--out", required=True, help="diretório de destino")
    ft.add_argument("--connections", type=int, default=8)
    ft.add_argument("--concurrent", type=int, default=4)
    ft.set_defaults(func=cmd_fetch)

    rn = sub.add_parser("run", help="executa o pipeline completo (download, poda, recuperação e conversão)")
    rn.add_argument("model", help="identificador do Hugging Face (namespace/nome), "
                    "URL do Hugging Face ou do GitHub, ou diretório local")
    rn.add_argument("-o", "--out", required=True, help="diretório da execução")
    rn.add_argument("--skip-checks", action="store_true",
                    help="ignora verificações de pré-requisitos de ambiente")
    rn.add_argument("--restart", action="store_true",
                    help="descarta o estado e os artefatos do diretório antes de começar")
    rn.add_argument("--discard-source-weights", action="store_true",
                    help="consome os pesos multimodais durante a extração, reduzindo o "
                         "pico de disco; exige novo download para refazer a etapa")
    rn.add_argument("--allow-oversized", action="store_true",
                    help="aceita alvo de parâmetros acima do teto de treino da máquina")
    _add_pipeline_args(rn)
    rn.set_defaults(func=cmd_run)

    ef = sub.add_parser("effort", help="explica os níveis de agressividade da destilação")
    ef.add_argument("--batch-size", type=int, default=2,
                    help="lote usado para estimar o disco dos logits (padrão: 2)")
    ef.add_argument("--no-anim", action="store_true", help="não anima os medidores")
    ef.set_defaults(func=cmd_effort)

    ex = sub.add_parser("extract",
                        help="extrai o decoder de texto de um modelo multimodal")
    ex.add_argument("model", help="identificador do Hugging Face (namespace/nome), "
                    "URL do Hugging Face ou do GitHub, ou diretório local")
    ex.add_argument("-o", "--out", required=True, help="diretório de destino")
    ex.add_argument("--discard-source-weights", action="store_true",
                    help="consome os shards de origem, reduzindo o pico de disco")
    ex.set_defaults(func=cmd_extract)

    st = sub.add_parser("status", help="exibe o estado e progresso de uma execução")
    st.add_argument("-o", "--out", required=True, help="diretório da execução")
    st.set_defaults(func=cmd_status)

    return p


def _falha(msg: str, hint: str | None = None, *, exc: BaseException | None = None) -> int:
    """Imprime a falha no formato do pacote, com traceback só sob AGUARDENTE_DEBUG."""
    print(f"\nerro: {msg}", file=sys.stderr)
    if hint:
        print(f"  → {hint}", file=sys.stderr)
    if exc is not None and os.environ.get(DEBUG_ENV):
        import traceback
        traceback.print_exception(exc, file=sys.stderr)
    return 1


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
    # As famílias abaixo são previsíveis e mereciam mensagem própria; sem elas
    # qualquer uma chegava ao usuário como traceback.
    except MemoryError as e:
        return _falha(
            "memória insuficiente para concluir a etapa",
            "Reduza --target-params, --batch-size ou --seq-len, aumente --grad-accum, "
            "e feche outros aplicativos antes de repetir.", exc=e)
    except ModuleNotFoundError as e:
        return _falha(
            f"dependência ausente: {e.name}",
            "Instale a stack completa com `uv pip install 'aguardente[pipeline]'` e "
            "confirme com `aguardente doctor`.", exc=e)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            return _falha("disco cheio durante a escrita",
                          "Libere espaço ou aponte -o para outro volume. O estado da "
                          "execução foi preservado para retomada.", exc=e)
        if e.errno in (errno.EACCES, errno.EPERM):
            return _falha(f"permissão negada: {e.filename or e}",
                          "Verifique as permissões do diretório de execução.", exc=e)
        return _falha(f"falha de entrada/saída: {e}",
                      "Verifique o caminho de destino e a conexão de rede.", exc=e)
    except Exception as e:  # noqa: BLE001 — último recurso, com traceback opcional
        return _falha(f"falha inesperada: {type(e).__name__}: {e}",
                      f"Defina {DEBUG_ENV}=1 e repita o comando para ver o traceback completo.",
                      exc=e)


if __name__ == "__main__":
    raise SystemExit(main())

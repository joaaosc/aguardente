"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import errno
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, requires
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
from .pipeline import (DISK_MARGIN, RunOptions, run_pipeline, suggested_batch_size,
                       training_dtype_bytes)
from .preflight import Status, blocking, check_pipeline_deps, check_python, run_all
from .probe import probe
from .source import resolve_source
from .state import ARTIFACT_DIRS as ARTIFACT_DIRS, discard_run_artifacts, run_lock
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

# ----------------------------------------------------------------- install


def _pipeline_requirements() -> list[str]:
    """Lê a extra ``pipeline`` dos metadados do pacote instalado."""
    try:
        declared = requires("aguardente") or []
    except PackageNotFoundError as exc:
        raise AguardenteError(
            "os metadados do aguardente não foram encontrados",
            hint="Instale o programa com `uv tool install aguardente` antes de instalar o pipeline.",
        ) from exc

    markers = ("extra == 'pipeline'", 'extra == "pipeline"')
    return [item.partition(";")[0].strip()
            for item in declared if any(marker in item for marker in markers)]


def cmd_install(args: argparse.Namespace) -> int:
    """Instala as dependências opcionais no ambiente do executável atual."""
    python = check_python()
    if python.status is not Status.OK:
        raise AguardenteError(
            f"Python {python.detail} não é compatível com o pipeline",
            hint=python.hint or "Reinstale com `uv tool install --python 3.12 aguardente`.",
        )

    uv = shutil.which("uv")
    if uv is None:
        raise AguardenteError(
            "o comando uv não foi encontrado",
            hint="Instale com `curl -LsSf https://astral.sh/uv/install.sh | sh`.",
        )

    dependencies = _pipeline_requirements()
    if not dependencies:
        raise AguardenteError(
            "a lista de dependências do pipeline está vazia",
            hint="Reinstale o aguardente a partir do repositório atualizado.",
        )

    command = [uv, "pip", "install", "--python", sys.executable, *dependencies]
    ui.title("Instalação do pipeline", f"Python {sys.version_info.major}.{sys.version_info.minor}")
    ui.blank()
    ui.explain("Instala as dependências no mesmo ambiente usado por este comando.", indent=0)
    ui.blank()
    ui.command(shlex.join(command), label="executando")
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        ui.error(
            "instalação não concluída",
            cause=f"O uv encerrou com código {completed.returncode}.",
            action="Revise a mensagem acima, corrija a causa e execute `aguardente install` novamente.",
        )
        return completed.returncode

    result = check_pipeline_deps(deep=True)
    if result.status is not Status.OK:
        ui.error(
            "instalação concluída, mas a stack não pôde ser importada",
            cause=result.detail,
            action="Execute `aguardente doctor --verbose` para inspecionar o ambiente.",
        )
        return 1

    ui.blank()
    ui.done("dependências do pipeline instaladas", hint=result.detail)

    if _e_ambiente_de_uv_tool():
        ui.blank()
        ui.note(
            "este é um ambiente `uv tool`, e o que foi instalado agora não entra no "
            "recibo da ferramenta: um `uv tool upgrade aguardente` reconstrói o "
            "ambiente a partir do recibo e remove estas dependências. Para gravá-las "
            "de forma durável, reinstale com "
            f"`uv tool install --python 3.12 --force {shlex.join(_com_extras(dependencies))}`."
        )

    faltando = [c for c in _requisitos_externos() if c.status is not Status.OK]
    if faltando:
        ui.blank()
        ui.note("o pipeline também depende de programas fora do Python, e estes ainda "
                "não estão prontos: "
                + "; ".join(f"{c.name} ({c.detail})" for c in faltando)
                + ". Execute `aguardente doctor` para o diagnóstico completo.")
    return 0


def _e_ambiente_de_uv_tool() -> bool:
    """Detecta se o executável atual vive num ambiente gerenciado por `uv tool`.

    Importa porque `uv pip install` nesse ambiente não é registrado no recibo da
    ferramenta: o próximo `uv tool upgrade` reconstrói tudo e descarta o que foi
    instalado por fora, devolvendo o usuário ao mesmo erro de dependência.
    """
    return "uv" in Path(sys.prefix).parts and "tools" in Path(sys.prefix).parts


def _com_extras(dependencies: list[str]) -> list[str]:
    """Comando equivalente que grava as dependências no recibo da ferramenta."""
    return [arg for dep in dependencies for arg in ("--with", dep)] + ["aguardente"]


def _requisitos_externos() -> list:
    """Verificações do ambiente que `aguardente install` não resolve sozinho."""
    from .preflight import check_arch, check_aria2, check_macos
    return [check_macos(), check_arch(), check_aria2()]


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
    # O `doctor` existe para diagnosticar: aqui vale pagar o import de cada
    # pacote para distinguir "ausente" de "presente mas quebrado". O `run` não
    # paga esse custo — a RAM importada fica residente pela execução inteira.
    results = run_all(deep_pipeline=True)
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
    ui.field("parâmetros do decoder", _fmt_params(b.total), note=f"{b.total:,}")
    if p.stored_params != b.total:
        ui.field("parâmetros em arquivo", _fmt_params(p.stored_params),
                 note="inclui cópias de pesos compartilhados e componentes extras")
    ui.field("tamanho original", _fmt_bytes(weights_bytes(p.stored_params, BPW_FP16)))
    if p.gated:
        ui.blank()
        ui.warn("modelo de acesso restrito",
                "Baixe o modelo autorizado com o cliente HF e use o diretório local.")

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
    # O mesmo dtype que o `run` vai usar. Calcular o teto com o padrão de 2
    # bytes fazia o `plan` anunciar um alvo 1,33x maior do que o `run` aceita
    # em máquina sem MPS — e o `run` abortava sobre o número que o `plan`
    # tinha acabado de sugerir.
    dtype_bytes = training_dtype_bytes(getattr(args, "device", None))
    teto = budget.max_params_for_training(dtype_bytes=dtype_bytes)
    ui.field("treinável até", _fmt_params(teto))
    ui.field("comprimido cabe até", _fmt_params(budget.max_params_for_inference()))

    if args.other_ram_gb is not None or args.other_disk_gb is not None:
        if args.other_ram_gb is None or args.other_disk_gb is None:
            raise AguardenteError(
                "--other-ram-gb e --other-disk-gb precisam ser usados juntos",
                hint="Informe os dois para avaliar a outra máquina.")
        _secao_outra_maquina(a, p, args.other_ram_gb, args.other_disk_gb,
                             args.target_params, effort.get(getattr(args, "effort", None)))

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
    lote = args.batch_size
    if lote is None:
        # A mesma conta que `run` fará, com as mesmas entradas: sem MPS o treino
        # cai para float32 e o lote cabível é outro. Estimar aqui com números
        # diferentes anunciaria um lote que a execução não usaria.
        lote = suggested_batch_size(p, budget, seq_len=nivel.seq_len,
                                    dtype_bytes=dtype_bytes, training=True)
        ui.field("lote sugerido pela RAM", str(lote))

    ui.header("Estimativa de recursos")
    ui.table(
        ["formato", "tamanho"],
        [("original (16 bits por peso)", _fmt_bytes(weights_bytes(final_params, BPW_FP16))),
         ("comprimido (~4,5 bits)", _fmt_bytes(weights_bytes(final_params, BPW_INT4_EMBED_FP16))),
         ("memória por 2.048 tokens", _fmt_bytes(kv_cache_bytes(final, 2048))),
         ("memória por 8.192 tokens", _fmt_bytes(kv_cache_bytes(final, 8192))),
         ("RAM estática estimada no treino", _fmt_bytes(training_bytes(final_params, dtype_bytes=dtype_bytes))),
         (f"logits em disco (--effort {nivel.name})", _fmt_bytes(nivel.logit_bytes()))],
        align_right=(1,),
    )

    _effort_panel(nivel, [], animate=not getattr(args, "no_anim", False))

    if plan and training_bytes(final_params, dtype_bytes=dtype_bytes) > budget.ram_bytes:
        ui.warn("o treino pode exceder a memória disponível",
                "Reduza o alvo de parâmetros ou execute a recuperação em uma máquina com mais RAM.")

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
                         target_params: int | None, nivel) -> None:
    """Avalia conversão e destilação numa máquina descrita só por RAM e disco.

    Não toca a máquina atual nem baixa nada: usa as mesmas contas do plano
    local (`Budget`, `plan_for_target`), só que sobre uma `Machine` hipotética.
    A pergunta que decide viabilidade de destilação não é "cabe o modelo na
    RAM", e sim "o piso de poda desta arquitetura cabe no teto de treino" —
    são números diferentes, e o segundo costuma ser bem menor.
    """
    m = Machine.other(ram_gb=ram_gb, disk_gb=disk_gb)
    budget = Budget.for_machine(m)
    # Aqui o padrão de 2 bytes é o correto, e não uma omissão: a outra máquina
    # é descrita só por RAM e disco, e todo Mac que roda este programa é Apple
    # Silicon — logo, treina em float16 sobre MPS.
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
    # Os logits pré-computados costumam ser o maior artefato do pipeline, e são
    # guardados por `_ensure_disk` como qualquer outra etapa. Omiti-los aqui
    # anunciava "conversão viável" para máquinas onde a execução real aborta.
    logits = nivel.logit_bytes()
    pico = download + extracao + poda + bundle + logits
    disco_viavel = disk_gb * GB >= pico * DISK_MARGIN

    ui.field("disco no pico (estimado)", _fmt_bytes(pico),
             note=f"download {_fmt_bytes(download)}"
                  + (f" + extração {_fmt_bytes(extracao)}" if extracao else "")
                  + f" + poda {_fmt_bytes(poda)} + bundle {_fmt_bytes(bundle)}"
                  + f" + logits {_fmt_bytes(logits)} (--effort {nivel.name})")

    ui.blank()
    ui.field("conversão (poda + export)", "viável" if disco_viavel else "inviável",
             note=None if disco_viavel else
             f"faltam ~{_fmt_bytes(pico * DISK_MARGIN - disk_gb * GB)} de disco")
    ui.field("destilação (recuperação)", "viável" if destilacao_viavel else "inviável",
             note=None if destilacao_viavel else
             "o piso de poda excede o teto de treino desta máquina — só "
             "--skip-recover chegaria a um resultado nela")


# ------------------------------------------------------------------ effort


def _effort_panel(nivel, sobrescritos, *, animate: bool = True) -> None:
    """Painel do nível escolhido: escala, o que muda e o que custa."""
    ui.header("Esforço da destilação")
    ui.meter_line(effort.index(nivel), total=len(effort.LEVELS),
                  label=f"{ui.bold(nivel.name)}  {ui.dim('· ' + nivel.label)}",
                  animate=animate)
    ui.explain("Valores efetivos para esta execução." if sobrescritos else nivel.summary)
    ui.blank()
    ui.field("custo relativo", f"{effort.relative_cost(nivel):.2f}× do nível "
                               f"{effort.DEFAULT}")
    ui.field("logits em disco", _fmt_bytes(nivel.logit_bytes()))
    ui.field("épocas de recuperação", str(nivel.epochs))
    ui.field("profundidade top-k", str(nivel.top_k))
    ui.field("amostras da cauda", str(nivel.tail_samples))
    if sobrescritos:
        ui.blank()
        ui.note(f"informado na linha de comando, com precedência sobre o preset: "
                f"{', '.join('--' + c.replace('_', '-') for c in sobrescritos)}")


def cmd_effort(args: argparse.Namespace) -> int:
    """Explica a escala de esforço sem tocar em nenhum modelo."""
    animar = not getattr(args, "no_anim", False)

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
         for l in effort.comparison()],
        align_right=(1, 2, 3, 4, 5, 6),
    )
    ui.blank()
    ui.explain(
        f"O nível fixa quantas amostras existem, então o disco não depende de "
        f"--batch-size: um lote maior processa as mesmas amostras em menos passos. "
        f"O custo é o tempo relativo ao nível {effort.DEFAULT}, calculado pelos "
        f"tokens processados em cada etapa.",
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
    if p.layout.is_multimodal:
        ui.field("descartado", _fmt_params(p.dropped_params),
                 note=f"{p.dropped_params / p.stored_params:.1%} do checkpoint")
    else:
        ui.field("descartado", "nada",
                 note="checkpoint de texto puro: só as chaves são renomeadas")

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
        student=args.student,
        connections=args.connections,
        concurrent=args.concurrent,
        batch_size=args.batch_size,
        calib_dataset=args.calib_dataset,
        calib_file=args.calib_file,
        assistant_only=args.assistant_only,
        calib_config=args.calib_config, calib_split=args.calib_split,
        text_column=args.text_column, eval_file=args.eval_file, seed=args.seed,
        reconstruct=not args.no_reconstruction, max_ppl_ratio=args.max_ppl_ratio,
        effort=nivel.name,
        no_checkpointing=args.no_checkpointing,
        recovery=args.recovery, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_targets=tuple(args.lora_targets.split(",")) if args.lora_targets else (),
        discard_source_weights=getattr(args, "discard_source_weights", False),
        **valores,
        platform=args.platform,
        compression=args.compression,
        compression_config=args.compression_config,
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
    with run_lock(out_dir):
        return discard_run_artifacts(out_dir)


def cmd_run(args: argparse.Namespace) -> int:
    """Executa o pipeline completo."""
    is_json = getattr(args, "json", False)
    # Em modo --json o stdout é o fluxo de eventos NDJSON: o aviso de
    # resolução, se houver, sai só como linha de comentário do relatório final.
    avisos: list[str] = []
    args.model = resolve_source(args.model,
                                report=(avisos.append if not is_json else (lambda _m: None)))
    opts = _options_from(args)
    effective_effort = replace(args.effort_level, **{k: getattr(opts, k) for k in effort.CONTROLLED})
    event_log = stdout_log() if is_json else None

    if not is_json:
        ui.title(f"aguardente · {opts.model}",
                 "Poda estruturada, destilação e conversão para Core AI")
        for aviso in avisos:
            ui.step(aviso)

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
            _effort_panel(effective_effort, args.effort_overrides)
        ui.blank()
        ui.rule()

    if is_json:
        nivel = effective_effort
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
        validation = ctx.state.output("export", "runtime_validation")
        if validation:
            ui.field("validação no runtime", str(validation))
        else:
            ui.field("runtime", "relatório de validação ausente")
    else:
        ui.done(f"pipeline concluído em {_fmt_seconds(total)}",
                hint=str(opts.out_dir))
    return 0


# ------------------------------------------------------------------ status


def cmd_status(args: argparse.Namespace) -> int:
    """Exibe o estado de uma execução existente."""
    from .state import RunState, run_is_active
    import json

    d = Path(args.out).expanduser()
    if not (d / "state.json").is_file():
        ui.error(f"nenhuma execução encontrada em {d}",
                 cause="O arquivo state.json não existe neste diretório.",
                 action="Verifique o caminho ou inicie uma execução com "
                        "`aguardente run <modelo> -o <diretório>`.")
        return 1

    active = run_is_active(d)
    st = RunState.load_or_create(d, reset_interrupted=not active)
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
    quality_path = d / "quality.json"
    if active:
        ui.note("há uma execução usando este diretório")
    elif quality_path.is_file() and not json.loads(quality_path.read_text()).get("passed"):
        ui.error("student reprovado no limite de qualidade", action=f"Consulte {quality_path}.")
        return 1
    elif pendentes:
        ui.note(f"etapas pendentes: {', '.join(pendentes)}")
        ui.note("para retomar, repita o comando original com as mesmas opções")
    else:
        ui.done("etapas registradas concluídas")
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
    p.add_argument("--student",
                   help="usa este modelo pronto como student, em vez de podar o "
                        "teacher — identificador do Hugging Face ou diretório local. "
                        "Precisa usar tokenizer equivalente ao do teacher")
    p.add_argument("--connections", type=int, default=8,
                   help="número de conexões por servidor no aria2c")
    p.add_argument("--concurrent", type=int, default=4,
                   help="número de downloads simultâneos no aria2c")
    p.add_argument("--calib-batches", type=int, help=PRESET)
    p.add_argument("--batch-size", type=int,
                   help="amostras por lote; restrição de memória, não de esforço. "
                        "Padrão: calculado pela RAM disponível")
    p.add_argument("--seq-len", type=int, help=PRESET)
    p.add_argument("--calib-dataset", help="dataset de calibração (namespace/name)")
    p.add_argument("--calib-file", help="TXT com documentos separados por linha em branco, ou JSONL com text/messages")
    p.add_argument("--calib-config", help="configuração do dataset Hugging Face")
    p.add_argument("--calib-split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--eval-file", help="TXT/JSONL de validação separado do treino")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reconstruction", action="store_true", help="desativa ajuste local das projeções após poda")
    p.add_argument("--max-ppl-ratio", type=float, default=1.2,
                   help="limite explícito de perplexidade student/teacher no teste (padrão 1.2)")
    p.add_argument("--logit-batches", type=int, help=PRESET)
    p.add_argument("--top-k", type=int, help=PRESET)
    p.add_argument("--tail-samples", type=int, help="amostras da cauda por token; padrão definido por effort")
    p.add_argument("--epochs", type=int, help=PRESET)
    p.add_argument("--lr", type=float, help=PRESET)
    p.add_argument("--alpha", type=float, help=PRESET)
    p.add_argument("--temperature", type=float, help=PRESET)
    p.add_argument("--grad-accum", type=int, help=PRESET)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--recovery", choices=["full", "lora"], default="full",
                   help="treina todos os pesos ou adaptadores de baixo posto com base congelada")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--assistant-only", action="store_true", help="supervisiona somente respostas em JSONL messages; exige prefixos estáveis do chat_template")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", default="", help="nomes/sufixos de camadas separados por vírgula; padrão: lineares não compartilhadas")
    p.add_argument("--platform", default="macOS",
                   choices=["macOS", "iOS", "watchOS", "visionOS", "tvOS"])
    compression = p.add_mutually_exclusive_group()
    compression.add_argument("--compression", default="auto", help="auto compara int4, int8 e sem quantização; ou escolha uma receita explícita")
    compression.add_argument("--compression-config", help="receita YAML do coreai-opt")
    p.add_argument("--compute-precision", default="float16")
    p.add_argument("--max-context-length", type=int)
    p.add_argument("--export-dry-run", action="store_true",
                   help="valida argumentos de exportação sem converter o modelo")
    p.add_argument("--device", help="dispositivo de execução (mps, cpu)")
    p.add_argument("--measure", action="store_true",
                   help="mantida por compatibilidade; qualidade e runtime são sempre validados")
    p.add_argument("--skip-recover", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--json", action="store_true",
                   help="emite eventos estruturados em formato NDJSON no stdout")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aguardente",
        description="Poda estruturada, destilação e conversão para Core AI.",
    )
    p.add_argument("--version", action="version", version=f"aguardente {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="identifica tarefa e capacidades sem exigir decoder causal")
    inspect_cmd.add_argument("model")
    inspect_cmd.add_argument("--task", default="auto", help="tarefa declarada quando o config é ambíguo")
    inspect_cmd.add_argument("--target-ram-gib", type=float, default=8)
    inspect_cmd.add_argument("--target-reserve-gib", type=float, default=4)
    inspect_cmd.add_argument("--context", type=int, default=2048)
    inspect_cmd.set_defaults(func=cmd_inspect)

    cp = sub.add_parser("compress", help="compara receitas e valida uma tarefa no Core AI nativo")
    cp.add_argument("model", help="checkpoint local; qualquer arquitetura pode fornecer --adapter")
    cp.add_argument("--data", required=True, help="dados de calibração, validação e teste")
    cp.add_argument("-o", "--out", required=True)
    cp.add_argument("--task", default="auto", choices=["auto", "causal-lm", "masked-lm", "classification", "custom"])
    cp.add_argument("--adapter", help="arquivo.py:factory local e confiável que fornece ModelTask")
    cp.add_argument("--recipes", default="w8,w4-block32,mixed4-8", help="receitas separadas por vírgula")
    cp.add_argument("--seq-len", type=int, default=64)
    cp.add_argument("--seed", type=int, default=42)
    cp.add_argument("--max-relative-rmse", type=float, default=0.05)
    cp.add_argument("--max-loss-ratio", type=float, default=1.05)
    cp.add_argument("--max-score-drop", type=float, default=0.02)
    cp.add_argument("--quality-policy", help="JSON com tolerâncias e pisos absolutos min_scores por tarefa")
    cp.add_argument("--max-asset-mib", type=float, help="limite do arquivo; não é memória residente")
    cp.add_argument("--target-ram-gib", type=float, default=8)
    cp.add_argument("--target-reserve-gib", type=float, default=4)
    cp.add_argument("--select", choices=["size", "latency", "fidelity"], default="size",
                    help="objetivo entre candidatos aprovados da fronteira de compromissos")
    cp.add_argument("--timeout", type=int, default=1800, help="limite de segundos por candidato")
    from .compress import cmd_compress
    cp.set_defaults(func=cmd_compress)

    pred = sub.add_parser("predict", help="executa um bundle de tarefa estática no Core AI")
    pred.add_argument("bundle", help="diretório com model.aimodel e interface.json")
    pred.add_argument("--inputs", required=True, help="NPZ com tensores já preparados e nomeados")
    pred.add_argument("-o", "--out", required=True, help="NPZ de saída")
    from .task_runtime import cmd_predict
    pred.set_defaults(func=cmd_predict)

    answers = sub.add_parser("score-answers", help="avalia respostas geradas contra referências verificadas")
    answers.add_argument("data", help="JSONL com answer e expected")
    answers.add_argument("--min-exact-match", type=float, default=0.8)
    answers.add_argument("--max-repetition", type=float, default=0.05)
    from .evaluation import cmd_score_answers
    answers.set_defaults(func=cmd_score_answers)

    ins = sub.add_parser("install", help="instala todas as dependências do pipeline")
    ins.set_defaults(func=cmd_install)

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
    pl.add_argument("--batch-size", type=int,
                    help="amostras por lote. Padrão: calculado pela RAM disponível")
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


def cmd_inspect(args):
    import json
    from .capabilities import inspect_model
    from .target import TargetProfile
    result = inspect_model(resolve_source(args.model), task=args.task)
    result["target"] = TargetProfile(args.target_ram_gib, args.target_reserve_gib,
                                     args.context).to_dict()
    machine = Machine.detect()
    result["preparation"] = {"ram_bytes": machine.ram_bytes, "free_disk_bytes": machine.free_disk_bytes}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _falha(msg: str, hint: str | None = None, *, exc: BaseException | None = None,
           args: argparse.Namespace | None = None) -> int:
    """Imprime a falha no formato do pacote, com traceback só sob AGUARDENTE_DEBUG."""
    if args is not None:
        _erro_como_evento(args, msg, hint)
    print(f"\nerro: {msg}", file=sys.stderr)
    if hint:
        print(f"  → {hint}", file=sys.stderr)
    if exc is not None and os.environ.get(DEBUG_ENV):
        import traceback
        traceback.print_exception(exc, file=sys.stderr)
    return 1


def _erro_como_evento(args: argparse.Namespace, msg: str, hint: str | None = None) -> None:
    """Em modo `--json`, uma falha precisa chegar como evento, não como texto.

    Quem consome o NDJSON — a interface macOS — só entende eventos. Uma
    mensagem em stderr vira ali uma linha de log solta, e o motivo real da
    falha desaparece atrás de "processo encerrou com código 1", justamente
    quando é a informação que o usuário mais precisa.
    """
    if not getattr(args, "json", False):
        return
    from .events import stdout_log
    stdout_log().error("pipeline", msg, hint=hint)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AguardenteError as e:
        _erro_como_evento(args, e.message, e.hint)
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
            "e feche outros aplicativos antes de repetir.", exc=e, args=args)
    except ModuleNotFoundError as e:
        return _falha(
            f"dependência ausente: {e.name}",
            "Instale a stack completa com `aguardente install` e "
            "confirme com `aguardente doctor`.", exc=e, args=args)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            return _falha("disco cheio durante a escrita",
                          "Libere espaço ou aponte -o para outro volume. O estado da "
                          "execução foi preservado para retomada.", exc=e, args=args)
        if e.errno in (errno.EACCES, errno.EPERM):
            return _falha(f"permissão negada: {e.filename or e}",
                          "Verifique as permissões do diretório de execução.", exc=e, args=args)
        return _falha(f"falha de entrada/saída: {e}",
                      "Verifique o caminho de destino e a conexão de rede.", exc=e, args=args)
    except Exception as e:  # noqa: BLE001 — último recurso, com traceback opcional
        return _falha(f"falha inesperada: {type(e).__name__}: {e}",
                      f"Defina {DEBUG_ENV}=1 e repita o comando para ver o traceback completo.",
                      exc=e, args=args)


if __name__ == "__main__":
    raise SystemExit(main())

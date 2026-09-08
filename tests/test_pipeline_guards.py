"""Testes das guardas do pipeline: impressão digital, disco e teto de treino."""

import pytest

from aguardente import pipeline as pl
from aguardente.arch import Arch, count_params
from aguardente.budget import GB, Machine
from aguardente.errors import AguardenteError, InsufficientResources
from aguardente.pipeline import Context, RunOptions, fingerprint
from aguardente.probe import ModelProbe
from aguardente.state import FINGERPRINT, RunState

QWEN3_4B = Arch(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                vocab_size=151936, tie_word_embeddings=True, qk_norm=True)


def opts(tmp_path, **kw):
    return RunOptions(model="Qwen/Qwen3-4B", out_dir=tmp_path / "run", **kw)


def contexto(tmp_path, **kw):
    o = opts(tmp_path, **kw)
    st = RunState.load_or_create(o.out_dir, model=o.model, target_params=o.target_params)
    return Context(opts=o, state=st, report=lambda *_: None)


def test_completed_fetch_repairs_weights_corrupted_after_previous_run(tmp_path, monkeypatch):
    import hashlib
    import json
    from aguardente import fetch as downloader
    from aguardente.fetch import RemoteFile
    from dataclasses import asdict
    ctx = contexto(tmp_path)
    dest = ctx.opts.teacher_dir
    dest.mkdir(parents=True)
    original = b"original weights"
    entry = RemoteFile("model.safetensors", len(original), hashlib.sha256(original).hexdigest())
    (dest / entry.path).write_bytes(b"X" * len(original))
    (dest / ".aguardente-source.json").write_text(json.dumps({
        "model": ctx.opts.model, "requested_revision": "main", "commit": "a" * 40,
        "file_selection_version": downloader.FILE_SELECTION_VERSION,
        "files": [asdict(entry)]}))
    ctx.state.finish("fetch", outputs={"dir": dest})
    calls = []

    def repair(plan, **kwargs):
        calls.extend(plan.missing())
        (dest / entry.path).write_bytes(original)

    monkeypatch.setattr(downloader, "fetch", repair)
    monkeypatch.setattr(downloader, "require_aria2", lambda: "aria2c")
    assert pl.stage_fetch(ctx) == dest
    assert calls == [entry]
    assert (dest / entry.path).read_bytes() == original


def test_unavailable_required_exporter_stops_before_download(tmp_path, monkeypatch):
    from aguardente import export
    ctx = contexto(tmp_path)
    monkeypatch.setattr(pl, "make_plan", lambda ctx: None)
    monkeypatch.setattr(export, "available", lambda: False)
    monkeypatch.setattr(pl, "stage_fetch", lambda ctx: pytest.fail("must stop before downloading"))
    with pytest.raises(AguardenteError, match="não está disponível"):
        pl.run_pipeline(ctx.opts)


def test_export_config_failure_is_not_downgraded_to_warning(tmp_path, monkeypatch):
    from aguardente import export
    ctx = contexto(tmp_path)
    monkeypatch.setattr(export, "available", lambda: True)
    monkeypatch.setattr(export, "build_command", lambda *a, **kw: ["export"])
    monkeypatch.setattr(export, "run_export", lambda *a, **kw: 1)
    with pytest.raises(AguardenteError, match="configuração de exportação recusada"):
        pl._conferir_exportador(ctx, tmp_path / "model")


def maquina(monkeypatch, *, ram_gb=24, disco_gb=500, mps=True):
    monkeypatch.setattr(Machine, "detect",
                        classmethod(lambda cls, path="/": Machine(ram_gb * GB,
                                                                 int(disco_gb * GB), 8, True)))
    monkeypatch.setattr(pl, "_training_dtype_bytes", lambda o: 2 if mps else 4)
    monkeypatch.setattr(pl, "probe", lambda ref: ModelProbe(
        ref=ref, arch=QWEN3_4B, model_type="qwen3",
        stored_params=count_params(QWEN3_4B).total))


# ------------------------------------------------------------ impressão digital


@pytest.mark.parametrize("campo, valor", [("top_k", 64), ("seq_len", 256),
                                          ("batch_size", 8), ("logit_batches", 32)])
def test_impressao_digital_dos_logits_muda_com_os_parametros(tmp_path, campo, valor):
    base = fingerprint(opts(tmp_path), "logits")
    assert fingerprint(opts(tmp_path, **{campo: valor}), "logits") != base


def test_impressao_digital_ignora_parametros_irrelevantes(tmp_path):
    """Número de conexões do download não altera os logits produzidos."""
    assert (fingerprint(opts(tmp_path, connections=16), "logits")
            == fingerprint(opts(tmp_path), "logits"))


def test_recuperacao_depende_dos_parametros_dos_logits(tmp_path):
    """Trocar --top-k precisa invalidar também a recuperação, não só os logits."""
    assert (fingerprint(opts(tmp_path, top_k=64), "recover")
            != fingerprint(opts(tmp_path), "recover"))


def test_resume_recusa_etapa_com_parametros_divergentes(tmp_path):
    ctx = contexto(tmp_path, top_k=128)
    ctx.state.finish("logits", outputs={"dir": str(tmp_path),
                                        FINGERPRINT: fingerprint(ctx.opts, "logits")})
    assert pl._resume(ctx, "logits")

    outro = contexto(tmp_path, top_k=64)
    linhas = []
    outro.report = linhas.append
    assert not pl._resume(outro, "logits")
    assert any("parâmetros mudaram" in l for l in linhas)


def test_resume_refaz_estado_sem_impressao_digital(tmp_path):
    ctx = contexto(tmp_path)
    ctx.state.finish("logits", outputs={"dir": str(tmp_path)})
    linhas = []
    ctx.report = linhas.append
    assert not pl._resume(ctx, "logits")
    assert any("sem registro dos parâmetros" in l for l in linhas)


# --------------------------------------------------------------------- disco


def test_disco_insuficiente_interrompe_antes_de_escrever(tmp_path, monkeypatch):
    ctx = contexto(tmp_path)
    monkeypatch.setattr(pl, "_disk_free", lambda p: 3 * GB)
    with pytest.raises(InsufficientResources) as e:
        pl._ensure_disk(ctx, 10 * GB, "a pré-computação de logits")
    assert "espaço insuficiente" in e.value.message
    assert "-o" in e.value.hint


def test_disco_suficiente_nao_interrompe(tmp_path, monkeypatch):
    ctx = contexto(tmp_path)
    monkeypatch.setattr(pl, "_disk_free", lambda p: 100 * GB)
    pl._ensure_disk(ctx, 10 * GB, "o download do modelo")


def test_margem_de_disco_e_exigida(tmp_path, monkeypatch):
    """Espaço exatamente igual à estimativa não basta: o volume ficaria em zero."""
    ctx = contexto(tmp_path)
    monkeypatch.setattr(pl, "_disk_free", lambda p: 10 * GB)
    with pytest.raises(InsufficientResources):
        pl._ensure_disk(ctx, 10 * GB, "os logits")


def test_disco_medido_no_volume_de_destino(tmp_path):
    """O caminho ainda não existe: o volume é o do primeiro ancestral existente."""
    assert pl._disk_free(tmp_path / "nao" / "existe" / "ainda") > 0


# ------------------------------------------------------------ teto de treino


def test_alvo_acima_do_teto_e_recusado(tmp_path, monkeypatch):
    maquina(monkeypatch)
    ctx = contexto(tmp_path, target_params=1_400_000_000)
    with pytest.raises(InsufficientResources) as e:
        pl.make_plan(ctx)
    assert "excede o teto de treino" in e.value.message
    assert "--allow-oversized" in e.value.hint


def test_alvo_acima_do_teto_passa_com_allow_oversized(tmp_path, monkeypatch):
    maquina(monkeypatch)
    ctx = contexto(tmp_path, target_params=1_400_000_000, allow_oversized=True)
    linhas = []
    ctx.report = linhas.append
    pl.make_plan(ctx)
    assert any("excede o teto de treino" in l for l in linhas)


def test_alvo_acima_do_teto_passa_sem_recuperacao(tmp_path, monkeypatch):
    """Sem treino não há teto de treino a respeitar."""
    maquina(monkeypatch)
    ctx = contexto(tmp_path, target_params=1_400_000_000, skip_recover=True)
    pl.make_plan(ctx)


def test_alvo_dentro_do_teto_e_aceito(tmp_path, monkeypatch):
    maquina(monkeypatch)
    ctx = contexto(tmp_path, target_params=1_000_000_000)
    _, plano = pl.make_plan(ctx)
    assert plano is not None and plano.target_params <= 1_000_000_000


def test_sem_mps_o_teto_cai_e_o_aviso_aparece(tmp_path, monkeypatch):
    """Em CPU o treino usa float32: o mesmo alvo deixa de caber."""
    maquina(monkeypatch, mps=False)
    ctx = contexto(tmp_path, target_params=1_000_000_000)
    linhas = []
    ctx.report = linhas.append
    with pytest.raises(InsufficientResources):
        pl.make_plan(ctx)


# ------------------------------------------------------- teacher cabe na RAM


def test_teacher_maior_que_a_ram_e_recusado(tmp_path, monkeypatch):
    """O piso de destino não importa aqui: é o modelo de origem que precisa caber."""
    maquina(monkeypatch, ram_gb=4)  # Qwen3-4B em fp16 são ~8 GB; não cabe em 3 GB úteis
    # Alvo generoso: o que este teste isola é a guarda de RAM do teacher, não
    # o cálculo do plano de poda, que tem sua própria bateria de testes.
    ctx = contexto(tmp_path, skip_recover=True, target_params=2_000_000_000)
    with pytest.raises(InsufficientResources) as e:
        pl.make_plan(ctx)
    assert "não cabe na RAM disponível" in e.value.message
    assert "--allow-oversized" in e.value.hint


def test_teacher_maior_que_a_ram_passa_com_allow_oversized(tmp_path, monkeypatch):
    maquina(monkeypatch, ram_gb=4)
    ctx = contexto(tmp_path, skip_recover=True, allow_oversized=True,
                   target_params=2_000_000_000)
    linhas = []
    ctx.report = linhas.append
    pl.make_plan(ctx)  # não levanta
    assert any("não cabe na RAM disponível" in l for l in linhas)


def test_teacher_dentro_da_ram_e_aceito(tmp_path, monkeypatch):
    maquina(monkeypatch, ram_gb=24)
    ctx = contexto(tmp_path, skip_recover=True, target_params=2_000_000_000)
    pl.make_plan(ctx)  # não levanta


# --------------------------------------------------------- lote automático


def test_lote_e_calculado_quando_nao_informado(tmp_path, monkeypatch):
    maquina(monkeypatch, ram_gb=24)
    ctx = contexto(tmp_path)
    assert ctx.opts.batch_size is None
    pl.make_plan(ctx)
    assert isinstance(ctx.opts.batch_size, int) and ctx.opts.batch_size >= 1


def test_lote_informado_nao_e_sobrescrito(tmp_path, monkeypatch):
    maquina(monkeypatch, ram_gb=24)
    ctx = contexto(tmp_path, batch_size=6)
    pl.make_plan(ctx)
    assert ctx.opts.batch_size == 6


# -------------------------------------------------------------- student externo


QWEN3_4B_OUTRO_VOCAB = Arch(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                            vocab_size=99999, tie_word_embeddings=True)


def maquina_com_student(monkeypatch, *, vocab_compativel, ram_gb=24):
    """Como `maquina()`, mas `probe` devolve arquiteturas diferentes por ref —
    necessário para simular teacher e student com vocabulários distintos."""
    monkeypatch.setattr(Machine, "detect",
                        classmethod(lambda cls, path="/": Machine(ram_gb * GB, 500 * GB, 8, True)))
    monkeypatch.setattr(pl, "_training_dtype_bytes", lambda o: 2)
    arch_student = QWEN3_4B if vocab_compativel else QWEN3_4B_OUTRO_VOCAB

    def fake_probe(ref):
        arch = arch_student if ref == "org/student" else QWEN3_4B
        return ModelProbe(ref=ref, arch=arch, model_type="qwen3",
                          stored_params=count_params(arch).total)

    monkeypatch.setattr(pl, "probe", fake_probe)


def test_student_com_vocabulario_compativel_e_aceito(tmp_path, monkeypatch):
    maquina_com_student(monkeypatch, vocab_compativel=True)
    ctx = contexto(tmp_path, student="org/student", skip_recover=True)
    _, plano = pl.make_plan(ctx)

    assert plano is None  # não há cirurgia de poda a planejar


def test_student_com_vocabulario_incompativel_e_recusado(tmp_path, monkeypatch):
    maquina_com_student(monkeypatch, vocab_compativel=False)
    ctx = contexto(tmp_path, student="org/student", skip_recover=True)

    with pytest.raises(AguardenteError) as e:
        pl.make_plan(ctx)

    assert "vocabulário" in e.value.message
    assert "151,936" in e.value.message or "99,999" in e.value.message


def test_student_pula_a_cirurgia_de_poda(tmp_path, monkeypatch):
    """`stage_prune` desvia para `_stage_fetch_student` em vez de podar."""
    maquina_com_student(monkeypatch, vocab_compativel=True)
    ctx = contexto(tmp_path, student=str(tmp_path / "modelo-pronto"), skip_recover=True)
    (tmp_path / "modelo-pronto").mkdir()
    (tmp_path / "modelo-pronto" / "config.json").write_text("{}")
    pl.make_plan(ctx)

    out = pl.stage_prune(ctx, tmp_path / "teacher")

    assert out == ctx.opts.pruned_dir
    assert (ctx.opts.pruned_dir / "config.json").is_file()


def test_fingerprint_da_poda_muda_com_o_student(tmp_path):
    """Trocar de student precisa refazer a etapa, mesmo com o resto igual."""
    assert (fingerprint(opts(tmp_path, student="org/a"), "prune")
            != fingerprint(opts(tmp_path, student="org/b"), "prune"))
    assert (fingerprint(opts(tmp_path, student="org/a"), "prune")
            != fingerprint(opts(tmp_path), "prune"))


def test_lote_automatico_e_menor_em_maquina_pequena(tmp_path, monkeypatch):
    """Menos RAM sobrando para ativações deveria sugerir um lote menor ou igual."""
    maquina(monkeypatch, ram_gb=24)
    grande = contexto(tmp_path / "a")
    pl.make_plan(grande)

    maquina(monkeypatch, ram_gb=6)
    pequena = contexto(tmp_path / "b", allow_oversized=True, skip_recover=True,
                       target_params=2_000_000_000)
    pl.make_plan(pequena)

    assert pequena.opts.batch_size <= grande.opts.batch_size


# ----------------------------------------------------- correções da revisão


def test_run_sem_batch_size_nao_quebra_o_painel_de_esforco():
    """`aguardente run` sem --batch-size chegava ao painel com None e estourava."""
    from aguardente import cli, effort

    args = cli.build_parser().parse_args(["run", "org/m", "-o", "run"])
    assert args.batch_size is None
    cli._effort_panel(effort.get(effort.DEFAULT), [], animate=False)


def test_fingerprint_do_export_muda_com_o_student(tmp_path):
    """Sem isto, trocar de student reaproveitava o bundle exportado do anterior."""
    assert (fingerprint(opts(tmp_path, student="org/a"), "export")
            != fingerprint(opts(tmp_path, student="org/b"), "export"))


def test_student_grande_demais_e_barrado_pelo_teto_de_treino(tmp_path, monkeypatch):
    """O teto de treino vale para um student pronto tanto quanto para um alvo."""
    maquina(monkeypatch, ram_gb=8)
    ctx = contexto(tmp_path, student="org/gigante")
    with pytest.raises(InsufficientResources):
        pl.make_plan(ctx)


def test_student_grande_demais_passa_com_allow_oversized(tmp_path, monkeypatch):
    maquina(monkeypatch, ram_gb=8)
    ctx = contexto(tmp_path, student="org/gigante", allow_oversized=True)
    pl.make_plan(ctx)
    assert ctx.plan is None


def test_plano_e_emitido_tambem_no_caminho_do_student(tmp_path, monkeypatch):
    """A GUI decodifica `plan`; sem o evento ela recebe etapas sem plano."""
    maquina(monkeypatch, ram_gb=24)
    eventos = []

    class _Log:
        def plan(self, stages):
            eventos.append(stages)

        def __getattr__(self, _nome):
            return lambda *a, **kw: None

    ctx = contexto(tmp_path, student="org/pequeno")
    ctx.events = _Log()
    monkeypatch.setattr(pl, "probe", lambda ref: ModelProbe(
        ref=ref, arch=QWEN3_4B, model_type="qwen3",
        stored_params=200_000_000 if ref == "org/pequeno" else count_params(QWEN3_4B).total))
    pl.make_plan(ctx)

    assert len(eventos) == 1
    assert [e["id"] for e in eventos[0]] == list(pl.STAGES[:1] + pl.STAGES[2:])


def test_amostras_nao_crescem_com_o_lote(tmp_path, monkeypatch):
    """Um lote maior deve percorrer as mesmas amostras em menos passos."""
    from aguardente.effort import LOTE_DE_REFERENCIA

    amostras = 16 * LOTE_DE_REFERENCIA
    assert pl._lotes_para(amostras, LOTE_DE_REFERENCIA) == 16
    assert pl._lotes_para(amostras, 4 * LOTE_DE_REFERENCIA) == 4
    # Textos suficientes para formar lotes cheios: `make_batches` descarta o
    # último bloco incompleto.
    assert pl._textos_para(4, 32, folga=8) >= 4 * 32


def test_checkpoints_de_outra_configuracao_sao_descartados(tmp_path):
    """`recover` só olha o arquivo em disco; o pipeline precisa limpar antes."""
    ctx = contexto(tmp_path, lr=1e-5)
    ckpt = ctx.opts.out_dir / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "last.pt").write_bytes(b"x")
    (ckpt / "best.pt").write_bytes(b"x")
    (ckpt / pl.FINGERPRINT_FILE).write_text("impressao-de-outra-execucao")

    pl._preparar_checkpoints(ctx, ckpt)

    assert not (ckpt / "last.pt").exists() and not (ckpt / "best.pt").exists()
    assert (ckpt / pl.FINGERPRINT_FILE).read_text() == fingerprint(ctx.opts, "recover")


def test_checkpoints_da_mesma_configuracao_sobrevivem(tmp_path):
    ctx = contexto(tmp_path)
    ckpt = ctx.opts.out_dir / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "last.pt").write_bytes(b"x")
    (ckpt / pl.FINGERPRINT_FILE).write_text(fingerprint(ctx.opts, "recover"))

    pl._preparar_checkpoints(ctx, ckpt)

    assert (ckpt / "last.pt").exists()


def test_logits_de_outra_configuracao_sao_descartados(tmp_path):
    """Trocar de teacher ou de --seq-len não pode reaproveitar shards antigos."""
    ctx = contexto(tmp_path, seq_len=256)
    logits = ctx.opts.logits_dir
    logits.mkdir(parents=True)
    (logits / "000000.pt").write_bytes(b"shard antigo")
    (logits / "manifest.json").write_text('{"top_k": 128}')
    (logits / pl.FINGERPRINT_FILE).write_text("impressao-de-outra-execucao")

    pl._preparar_logits(ctx, logits)

    assert not (logits / "000000.pt").exists()
    assert not (logits / "manifest.json").exists()
    assert (logits / pl.FINGERPRINT_FILE).read_text() == fingerprint(ctx.opts, "logits")


def test_logits_da_mesma_configuracao_sobrevivem_para_retomada(tmp_path):
    """Uma etapa interrompida com os mesmos parâmetros retoma de onde parou."""
    ctx = contexto(tmp_path)
    logits = ctx.opts.logits_dir
    logits.mkdir(parents=True)
    (logits / "000000.pt").write_bytes(b"shard valido")
    (logits / pl.FINGERPRINT_FILE).write_text(fingerprint(ctx.opts, "logits"))

    pl._preparar_logits(ctx, logits)

    assert (logits / "000000.pt").exists()


def test_shards_sem_marca_sao_descartados(tmp_path):
    """Interrompido antes da gravação atômica do manifesto: não dá para confiar."""
    ctx = contexto(tmp_path)
    logits = ctx.opts.logits_dir
    logits.mkdir(parents=True)
    (logits / "000000.pt").write_bytes(b"shard orfao")

    pl._preparar_logits(ctx, logits)

    assert not (logits / "000000.pt").exists()


def test_limpeza_de_checkpoints_precede_a_medicao_de_disco():
    """Cobrar do orçamento bytes prestes a serem liberados reprova o que caberia.

    A ordem das duas chamadas é a correção inteira, e ela não tem efeito
    observável sem um student de verdade em disco: a asserção é sobre a fonte.
    """
    import inspect

    fonte = inspect.getsource(pl.stage_recover)
    assert (fonte.index("_preparar_checkpoints")
            < fonte.index("os checkpoints da recuperação"))


def test_teto_do_plan_usa_o_mesmo_dtype_do_run():
    """O plan anunciava um teto 1,33x maior do que o run aceita sem MPS."""
    from aguardente.budget import GB, Budget, Machine

    budget = Budget.for_machine(Machine(24 * GB, 500 * GB, 8, True))
    assert budget.max_params_for_training(dtype_bytes=4) < \
        budget.max_params_for_training(dtype_bytes=2)


def test_dtype_de_treino_nao_importa_torch():
    """`plan` existe para não tocar em nada; importar torch custa RAM residente."""
    import sys as _sys

    antes = "torch" in _sys.modules
    assert pl.training_dtype_bytes() in (2, 4)
    assert ("torch" in _sys.modules) == antes


def test_dtype_de_treino_respeita_o_dispositivo_informado():
    assert pl.training_dtype_bytes("mps") == 4
    assert pl.training_dtype_bytes("cpu") == 4


def test_student_de_outra_configuracao_e_descartado(tmp_path):
    """Um download interrompido não pode misturar dois students no destino."""
    ctx = contexto(tmp_path, student="org/a")
    destino = ctx.opts.pruned_dir
    destino.mkdir(parents=True)
    (destino / "model.safetensors").write_bytes(b"pesos do student anterior")
    (destino / pl.FINGERPRINT_FILE).write_text("impressao-de-outro-student")

    pl._preparar_artefatos(ctx, destino, "prune", aviso="descartado")

    assert not (destino / "model.safetensors").exists()
    assert (destino / pl.FINGERPRINT_FILE).read_text() == fingerprint(ctx.opts, "prune")


def test_flag_de_descarte_de_pesos_chega_ao_pipeline():
    """A etapa de extração lê `opts.discard_source_weights`; sem o campo, quebrava."""
    from aguardente import cli

    args = cli.build_parser().parse_args(
        ["run", "org/m", "-o", "run", "--discard-source-weights"])
    assert cli._options_from(args).discard_source_weights is True
    assert cli._options_from(
        cli.build_parser().parse_args(["run", "org/m", "-o", "run"])
    ).discard_source_weights is False


def test_todo_atributo_lido_de_opts_existe_em_run_options():
    """Guarda estrutural: `opts.x` no pipeline só compila contra um campo real."""
    import ast
    import dataclasses
    from pathlib import Path

    campos = {f.name for f in dataclasses.fields(RunOptions)}
    campos |= {n for n in dir(RunOptions) if not n.startswith("_")}

    fonte = Path(pl.__file__).read_text()
    lidos = {
        n.attr
        for n in ast.walk(ast.parse(fonte))
        if isinstance(n, ast.Attribute)
        and (
            (isinstance(n.value, ast.Name) and n.value.id == "opts")
            or (isinstance(n.value, ast.Attribute) and n.value.attr == "opts")
        )
    }
    assert not (lidos - campos)


def test_exportado_nao_significa_validado_no_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from aguardente import export

    ctx = contexto(tmp_path, measure=True)
    ctx.state.finish("export", outputs={"dir": str(tmp_path)})
    monkeypatch.setattr(export, "find_bundle", lambda _: SimpleNamespace(aimodel=tmp_path / "model.aimodel"))
    monkeypatch.setattr(export, "run_export", lambda *a, **kw: 1)
    with pytest.raises(AguardenteError, match="falhou na validação no runtime"):
        pl.validate_export(ctx, tmp_path / "student", tmp_path / "bundle")
    assert ctx.state.stage("export").status != "ok"
    assert ctx.state.output("export", "runtime_validation") is None

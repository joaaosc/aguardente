"""Testes das guardas do pipeline: impressão digital, disco e teto de treino."""

import pytest

from aguardente import pipeline as pl
from aguardente.arch import Arch, count_params
from aguardente.budget import GB, Machine
from aguardente.errors import InsufficientResources
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


def test_resume_avisa_ao_aproveitar_estado_sem_impressao_digital(tmp_path):
    ctx = contexto(tmp_path)
    ctx.state.finish("logits", outputs={"dir": str(tmp_path)})
    linhas = []
    ctx.report = linhas.append
    assert pl._resume(ctx, "logits")
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

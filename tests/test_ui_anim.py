"""Testes dos indicadores animados do terminal.

Sob pytest a saída não é um TTY, então o caminho exercitado por padrão é o
degradado — que é justamente o que precisa continuar legível em log e em CI.
O caminho animado é testado forçando o interruptor.
"""

import pytest

from aguardente import pipeline as pl
from aguardente import ui


@pytest.fixture
def animado(monkeypatch):
    """Força o caminho animado sem depender de um terminal de verdade."""
    monkeypatch.setattr(ui, "animations_enabled", lambda: True)


# ------------------------------------------------------------- interruptor


def test_sem_tty_nao_anima():
    assert ui.animations_enabled() is False


def test_variavel_desliga_a_animacao(monkeypatch, animado):
    monkeypatch.setenv(ui.ANIM_ENV, "1")
    # `animations_enabled` foi substituída pela fixture; a original é quem lê a
    # variável, então a checagem é feita nela.
    monkeypatch.undo()
    monkeypatch.setenv(ui.ANIM_ENV, "1")
    assert ui.animations_enabled() is False


# ----------------------------------------------------------------- medidor


def test_medidor_distingue_niveis_sem_cor():
    """Cor não pode ser o único canal: em pipe ou sem cor a leitura continua."""
    niveis = [ui.meter(i, total=4) for i in range(4)]
    assert niveis == ["▁···", "▁▃··", "▁▃▅·", "▁▃▅▇"]
    assert len(set(niveis)) == 4


def test_medidor_estatico_quando_nao_anima(capsys):
    ui.meter_line(1, total=4, label="medium", indent=2)
    saida = capsys.readouterr().out
    assert saida == "  ▁▃··  medium\n"


def test_medidor_animado_desenha_um_quadro_por_nivel(capsys, animado):
    ui.meter_line(3, total=4, label="max")
    quadros = [q for q in capsys.readouterr().out.split("\r") if q.strip()]
    assert len(quadros) == 4
    assert quadros[0].endswith("▁···  max")
    assert quadros[-1].endswith("▁▃▅▇  max\n")


# ----------------------------------------------------------------- spinner


def test_spinner_sem_tty_imprime_linha_estatica(capsys):
    with ui.Spinner("carregando o modelo", indent=4):
        pass
    assert capsys.readouterr().out == "    carregando o modelo\n"


def test_spinner_sem_tty_nao_cria_thread():
    s = ui.Spinner("trabalhando")
    with s:
        assert s._thread is None


def test_spinner_animado_encerra_com_marca_de_sucesso(capsys, animado):
    with ui.Spinner("medindo importância", indent=2):
        pass
    saida = capsys.readouterr().out
    assert saida.rstrip().endswith("✓ medindo importância")


def test_spinner_animado_marca_falha_quando_o_bloco_levanta(capsys, animado):
    with pytest.raises(RuntimeError):
        with ui.Spinner("treinando"):
            raise RuntimeError("estourou")
    assert "✗ treinando" in capsys.readouterr().out


def test_spinner_troca_a_mensagem(capsys, animado):
    with ui.Spinner("primeira") as s:
        s.update("segunda")
    assert "✓ segunda" in capsys.readouterr().out


# ------------------------------------------------------------------- barra


def test_barra_preenche_proporcionalmente():
    b = ui.Progress(10, label="logits", width=10)
    b.update(5)
    assert b.fraction == 0.5
    assert b.bar().count(ui.BAR_FULL) == 5


def test_barra_completa_nao_deixa_trilho():
    b = ui.Progress(4, width=8)
    b.update(4)
    assert b.bar() == ui.BAR_FULL * 8


def test_barra_nao_ultrapassa_o_total():
    b = ui.Progress(3)
    b.update(99)
    assert b.current == 3 and b.fraction == 1.0


def test_barra_sem_tty_emite_marcos_a_cada_dez_por_cento(capsys):
    b = ui.Progress(100, label="baixando", indent=2)
    for i in range(1, 101):
        b.update(i)
    linhas = capsys.readouterr().out.strip().splitlines()
    assert len(linhas) == 11          # a primeira atualização e cada dezena
    assert "(1/100)" in linhas[0]
    assert "baixando 100%" in linhas[-1]
    assert all("%" in linha for linha in linhas)


def test_barra_animada_redesenha_no_lugar(capsys, animado):
    b = ui.Progress(4, label="logits")
    for i in range(1, 5):
        b.update(i)
    saida = capsys.readouterr().out
    assert "\n" not in saida          # tudo na mesma linha
    assert saida.count("\r") == 4


def test_barra_encerrada_resume_o_trabalho(capsys, animado):
    b = ui.Progress(40, label="pré-computando")
    b.update(40)
    b.done(suffix="40 shards")
    assert "✓ pré-computando" in capsys.readouterr().out


def test_avanco_incremental(capsys):
    b = ui.Progress(3, label="x")
    b.advance()
    b.advance(2)
    assert b.current == 3


# ------------------------------------------- integração com o pipeline


def contexto(tmp_path, *, animate):
    from aguardente.pipeline import Context, RunOptions
    from aguardente.state import RunState

    o = RunOptions(model="org/m", out_dir=tmp_path / "run")
    st = RunState.load_or_create(o.out_dir, model=o.model)
    linhas: list[str] = []
    return Context(opts=o, state=st, report=linhas.append, animate=animate), linhas


def test_atividade_sem_animacao_reporta_uma_vez(tmp_path):
    """Sob --json a linha animada viraria lixo no meio do NDJSON."""
    ctx, linhas = contexto(tmp_path, animate=False)
    with pl._Activity(ctx, "carregando") as a:
        a.update("medindo")
    assert linhas == [" " * pl.INDENT + "carregando"]


def test_progresso_sem_animacao_ainda_emite_eventos(tmp_path):
    ctx, _ = contexto(tmp_path, animate=False)
    emitidos = []
    ctx.events.progress = lambda sid, cur, tot=None: emitidos.append((sid, cur, tot))

    barra = pl._Progress(ctx, 10, label="logits", sid="logits")
    for i in range(1, 11):
        barra.update(i)

    # Coalescido por tempo: o primeiro e o último são garantidos, o meio não.
    assert emitidos[0] == ("logits", 1, 10)
    assert emitidos[-1] == ("logits", 10, 10)
    assert len(emitidos) < 10


def test_filtro_do_aria2_alimenta_a_barra_quando_existe(tmp_path):
    ctx, linhas = contexto(tmp_path, animate=False)
    barra = pl._Progress(ctx, 100, label="baixando", sid="fetch")
    filtro = pl._ProgressFilter(ctx.report, bar=barra)
    filtro("[#a1b2 1.2GiB/3.4GiB(35%) CN:8]")
    assert barra._current == 35
    assert linhas == []               # a barra é dona da saída, não o report


def test_filtro_sem_barra_mantem_o_comportamento_textual(tmp_path):
    ctx, linhas = contexto(tmp_path, animate=False)
    filtro = pl._ProgressFilter(ctx.report)
    filtro("[#a1b2 1.2GiB/3.4GiB(35%) CN:8]")
    assert linhas == ["              35%"]

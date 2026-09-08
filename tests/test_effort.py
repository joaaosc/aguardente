"""Testes dos níveis de esforço da destilação."""

import pytest

from aguardente import cli, effort
from aguardente.distill.teacher import estimate_logit_bytes
from aguardente.errors import AguardenteError

# Hiperparâmetros de referência; a cauda amostrada passa a fazer parte do
# objetivo de destilação e do orçamento explícito de armazenamento.
PADROES_ANTERIORES = dict(calib_batches=32, seq_len=512, logit_batches=256,
                          top_k=128, epochs=2, lr=3e-5, alpha=0.9,
                          temperature=2.0, grad_accum=4, tail_samples=128)


# ------------------------------------------------------------------ presets


def test_nivel_medio_reproduz_os_padroes_anteriores():
    assert effort.get("medium").values() == PADROES_ANTERIORES


def test_padrao_e_o_nivel_medio():
    assert effort.DEFAULT == "medium"
    assert effort.get(None).name == "medium"


def test_todos_os_niveis_definem_todas_as_opcoes():
    for nivel in effort.LEVELS:
        assert set(nivel.values()) == set(effort.CONTROLLED)
        assert all(v is not None for v in nivel.values().values())


@pytest.mark.parametrize("campo", ["calib_batches", "seq_len", "logit_batches",
                                   "top_k", "epochs"])
def test_escala_cresce_monotonicamente(campo):
    valores = [getattr(n, campo) for n in effort.LEVELS]
    assert valores == sorted(valores)
    assert valores[0] < valores[-1]


def test_taxa_de_aprendizado_cai_conforme_o_esforco_sobe():
    """Mais épocas com passo grande desestabiliza; o passo encolhe junto."""
    taxas = [n.lr for n in effort.LEVELS]
    assert taxas == sorted(taxas, reverse=True)


def test_custo_relativo_ordena_os_niveis():
    custos = [effort.relative_cost(n) for n in effort.LEVELS]
    assert custos == sorted(custos)
    assert effort.relative_cost(effort.get("medium")) == 1.0
    assert effort.relative_cost(effort.get("low")) < 1.0
    assert effort.relative_cost(effort.get("max")) > 5


def test_indice_segue_a_ordem_da_escala():
    assert [effort.index(n) for n in effort.LEVELS] == [0, 1, 2, 3]


def test_nome_desconhecido_gera_erro_acionavel():
    with pytest.raises(AguardenteError) as e:
        effort.get("turbo")
    assert "turbo" in e.value.message
    assert all(nome in e.value.hint for nome in effort.NAMES)


def test_nome_aceita_espaco_e_maiuscula():
    assert effort.get(" HIGH ").name == "high"


# -------------------------------------------------------------- estimativas


def test_disco_dos_logits_bate_com_a_estimativa_do_teacher():
    nivel = effort.get("high")
    esperado, _ = estimate_logit_bytes(nivel.logit_samples, nivel.seq_len,
                                       top_k=nivel.top_k, tail_samples=nivel.tail_samples)
    assert nivel.logit_bytes() == esperado


def test_disco_cresce_com_o_nivel():
    tamanhos = [n.logit_bytes() for n in effort.LEVELS]
    assert tamanhos == sorted(tamanhos)


def test_amostras_derivam_do_lote_de_referencia():
    """O nível fixa quantas amostras existem; o lote só decide como agrupá-las."""
    nivel = effort.get("medium")
    assert nivel.logit_samples == nivel.logit_batches * effort.LOTE_DE_REFERENCIA
    assert nivel.calib_samples == nivel.calib_batches * effort.LOTE_DE_REFERENCIA


def test_comparacao_cobre_todos_os_niveis():
    linhas = list(effort.comparison())
    assert [l["name"] for l in linhas] == list(effort.NAMES)
    assert all(l["logit_bytes"] > 0 for l in linhas)


# --------------------------------------------------------------- precedência


def test_preset_preenche_o_que_nao_foi_informado():
    valores, sobrescritos = effort.resolve(effort.get("high"),
                                           dict.fromkeys(effort.CONTROLLED))
    assert valores == effort.get("high").values()
    assert sobrescritos == []


def test_opcao_informada_vence_o_preset():
    """Deixar o preset sobrescrever a escolha explícita repetiria o defeito do estado."""
    informado = dict.fromkeys(effort.CONTROLLED)
    informado["epochs"] = 7
    valores, sobrescritos = effort.resolve(effort.get("low"), informado)
    assert valores["epochs"] == 7
    assert valores["top_k"] == effort.get("low").top_k
    assert sobrescritos == ["epochs"]


def test_campo_fora_do_preset_e_ignorado():
    valores, sobrescritos = effort.resolve(effort.get("low"), {"batch_size": 8})
    assert "batch_size" not in valores and sobrescritos == []


# ------------------------------------------------------------- integração cli


def _run_args(*extra):
    return cli.build_parser().parse_args(["run", "org/m", "-o", "run", *extra])


def test_cli_aplica_o_preset_escolhido():
    opts = cli._options_from(_run_args("--effort", "high"))
    alto = effort.get("high")
    assert opts.effort == "high"
    assert opts.epochs == alto.epochs and opts.top_k == alto.top_k
    assert opts.seq_len == alto.seq_len


def test_cli_sem_flag_mantem_o_comportamento_anterior():
    opts = cli._options_from(_run_args())
    for campo, valor in PADROES_ANTERIORES.items():
        assert getattr(opts, campo) == valor


def test_cli_registra_as_opcoes_sobrescritas():
    args = _run_args("--effort", "max", "--epochs", "1", "--top-k", "32")
    opts = cli._options_from(args)
    assert opts.epochs == 1 and opts.top_k == 32
    assert args.effort_overrides == ["epochs", "top_k"]
    assert opts.seq_len == effort.get("max").seq_len


def test_cli_recusa_nivel_invalido():
    with pytest.raises(SystemExit):
        _run_args("--effort", "turbo")


def test_esforco_participa_da_impressao_digital():
    """Trocar de nível muda os parâmetros, e o trabalho gravado deixa de valer."""
    from aguardente.pipeline import fingerprint

    baixo = cli._options_from(_run_args("--effort", "low"))
    alto = cli._options_from(_run_args("--effort", "high"))
    assert fingerprint(baixo, "logits") != fingerprint(alto, "logits")
    assert fingerprint(baixo, "recover") != fingerprint(alto, "recover")


def test_plan_tambem_aceita_o_nivel():
    args = cli.build_parser().parse_args(["plan", "org/m", "--effort", "low"])
    assert args.effort == "low"


def test_subcomando_effort_existe():
    args = cli.build_parser().parse_args(["effort"])
    assert args.func is cli.cmd_effort


def test_subcomando_effort_executa(capsys):
    assert cli.cmd_effort(cli.build_parser().parse_args(["effort", "--no-anim"])) == 0
    saida = capsys.readouterr().out
    for nivel in effort.LEVELS:
        assert nivel.name in saida and nivel.label in saida
    # A explicação passa por textwrap: comparar com os espaços normalizados.
    corrido = " ".join(saida.split())
    assert "--target-params decide o tamanho do resultado" in corrido
    assert "--effort decide o cuidado" in corrido

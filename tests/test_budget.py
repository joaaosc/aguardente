"""Testes de `suggest_batch_size`: escolha de lote a partir da RAM disponível."""

import pytest

from aguardente.budget import GB, suggest_batch_size


def test_maquina_generosa_usa_o_teto():
    lote = suggest_batch_size(hidden_size=2560, seq_len=512, ram_bytes=100 * GB)
    assert lote == 32  # o máximo, não um número inventado


def test_sem_margem_cai_no_minimo():
    """Pesos já ocupam tudo: nem lote 1 tem folga, mas o piso é sempre 1."""
    lote = suggest_batch_size(hidden_size=2560, seq_len=512, ram_bytes=1 * GB,
                              reserved_bytes=1 * GB)
    assert lote == 1


def test_treino_pede_lote_menor_que_inferencia():
    """O autograd retém mais buffers que um forward puro; o fator reflete isso."""
    kwargs = dict(hidden_size=4096, seq_len=1024, ram_bytes=6 * GB)
    lote_treino = suggest_batch_size(**kwargs, training=True)
    lote_inferencia = suggest_batch_size(**kwargs, training=False)
    assert lote_treino <= lote_inferencia


def test_sequencia_mais_longa_pede_lote_menor():
    kwargs = dict(hidden_size=2560, ram_bytes=6 * GB)
    curto = suggest_batch_size(seq_len=256, **kwargs)
    longo = suggest_batch_size(seq_len=2048, **kwargs)
    assert longo <= curto


def test_reservado_reduz_a_margem():
    kwargs = dict(hidden_size=2560, seq_len=512, ram_bytes=8 * GB)
    sem_reserva = suggest_batch_size(reserved_bytes=0, **kwargs)
    com_reserva = suggest_batch_size(reserved_bytes=6 * GB, **kwargs)
    assert com_reserva <= sem_reserva


def test_reservado_acima_da_ram_nao_fica_negativo():
    lote = suggest_batch_size(hidden_size=2560, seq_len=512, ram_bytes=4 * GB,
                              reserved_bytes=10 * GB)
    assert lote == 1


@pytest.mark.parametrize("minimo,maximo", [(1, 4), (2, 2), (1, 64)])
def test_respeita_o_intervalo_pedido(minimo, maximo):
    lote = suggest_batch_size(hidden_size=2560, seq_len=512, ram_bytes=1000 * GB,
                              minimum=minimo, maximum=maximo)
    assert minimo <= lote <= maximo


def test_unknown_ram_is_not_a_zero_gigabyte_training_budget():
    from aguardente.budget import Budget, Machine
    from aguardente.errors import InsufficientResources

    with pytest.raises(InsufficientResources, match="não foi possível medir a RAM"):
        Budget.for_machine(Machine(0, 100 * GB, 1, True))

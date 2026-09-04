"""Testes das guardas de reaproveitamento de diretório de execução."""

import json
import os

import pytest

from aguardente.errors import StateMismatch
from aguardente.state import FINGERPRINT, RunState, run_lock

MODELO = "Qwen/Qwen3-4B"
OUTRO = "HuggingFaceTB/SmolLM2-135M"


def criar(tmp_path, **kw):
    return RunState.load_or_create(tmp_path / "run", model=MODELO,
                                   target_params=1_000_000_000, **kw)


# ------------------------------------------------------- divergência de estado


def test_modelo_divergente_aborta(tmp_path):
    """Regressão: o modelo gravado tinha precedência e o pipeline seguia com o anterior."""
    criar(tmp_path).finish("fetch", outputs={"dir": str(tmp_path)})
    with pytest.raises(StateMismatch) as e:
        RunState.load_or_create(tmp_path / "run", model=OUTRO)
    assert MODELO in e.value.message and OUTRO in e.value.message
    assert "--restart" in e.value.hint


def test_alvo_divergente_aborta(tmp_path):
    criar(tmp_path).begin("fetch")
    with pytest.raises(StateMismatch) as e:
        RunState.load_or_create(tmp_path / "run", model=MODELO, target_params=900_000_000)
    assert "alvo" in e.value.message


def test_retomada_sem_argumentos_herda_o_gravado(tmp_path):
    """`aguardente status` carrega o estado sem informar modelo nem alvo."""
    criar(tmp_path).begin("fetch")
    st = RunState.load_or_create(tmp_path / "run")
    assert st.model == MODELO and st.target_params == 1_000_000_000


def test_retomada_com_os_mesmos_valores_e_aceita(tmp_path):
    criar(tmp_path).finish("fetch", outputs={"dir": str(tmp_path)})
    st = criar(tmp_path)
    assert st.is_done("fetch", require=("dir",))


def test_restart_descarta_o_estado_anterior(tmp_path):
    criar(tmp_path).finish("fetch", outputs={"dir": str(tmp_path)})
    st = RunState.load_or_create(tmp_path / "run", model=OUTRO, restart=True)
    assert st.model == OUTRO
    assert not st.stage("fetch").done


def test_estado_corrompido_vira_erro_acionavel(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "state.json").write_text("{isto não é json")
    with pytest.raises(StateMismatch) as e:
        RunState.load_or_create(d, model=MODELO)
    assert "corrompido" in e.value.message


def test_estado_com_json_valido_mas_nao_objeto(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "state.json").write_text("[1, 2, 3]")
    with pytest.raises(StateMismatch):
        RunState.load_or_create(d, model=MODELO)


# ------------------------------------------------------------ impressão digital


def test_impressao_digital_divergente_invalida_a_etapa(tmp_path):
    """Regressão: logits gerados com outro --top-k eram reaproveitados em silêncio."""
    st = criar(tmp_path)
    st.finish("logits", outputs={"dir": str(tmp_path), FINGERPRINT: "abc123"})
    assert st.is_done("logits", require=("dir",), fingerprint="abc123")
    assert not st.is_done("logits", require=("dir",), fingerprint="outra")


def test_etapa_sem_impressao_digital_e_aproveitada(tmp_path):
    """Estado de uma versão anterior não força refazer horas de trabalho."""
    st = criar(tmp_path)
    st.finish("logits", outputs={"dir": str(tmp_path)})
    assert st.fingerprint_of("logits") is None
    assert st.is_done("logits", require=("dir",), fingerprint="qualquer")


def test_impressao_digital_persiste_entre_cargas(tmp_path):
    st = criar(tmp_path)
    st.finish("prune", outputs={"dir": str(tmp_path), FINGERPRINT: "dedo"})
    assert criar(tmp_path).fingerprint_of("prune") == "dedo"


# --------------------------------------------------------------------- lock


def test_lock_impede_execucao_concorrente(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(StateMismatch) as e:
            with run_lock(tmp_path):
                pass
    assert "outra execução" in e.value.message


def test_lock_e_liberado_ao_sair(tmp_path):
    with run_lock(tmp_path) as lock:
        assert lock.exists()
    assert not lock.exists()


def test_lock_liberado_mesmo_com_excecao(tmp_path):
    with pytest.raises(RuntimeError):
        with run_lock(tmp_path):
            raise RuntimeError("falha na etapa")
    assert not (tmp_path / "run.lock").exists()


def test_lock_de_processo_morto_e_recuperado(tmp_path):
    """Um processo interrompido não pode deixar o diretório travado para sempre."""
    import socket
    import time
    (tmp_path / "run.lock").write_text(json.dumps(
        {"pid": 2 ** 30, "at": time.time(), "host": socket.gethostname()}))
    with run_lock(tmp_path) as lock:
        assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_lock_antigo_e_considerado_obsoleto(tmp_path):
    (tmp_path / "run.lock").write_text(json.dumps(
        {"pid": os.getpid(), "at": 0, "host": "outra-maquina"}))
    with run_lock(tmp_path, stale_after=10) as lock:
        assert lock.exists()

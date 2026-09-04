"""Testes de persistência de estado e retomada."""

import json

from aguardente.state import RunState, StageStatus


def test_state_file_materializes_on_first_stage(tmp_path):
    st = RunState.load_or_create(tmp_path / "run", model="org/m", target_params=1000)
    # Antes da primeira etapa o diretório não fica reservado em nome do modelo:
    # uma execução que falha no plano pode ser repetida com outro identificador
    # sem exigir --restart.
    assert not st.path.exists()

    st.begin("fetch")
    assert st.path.is_file()
    assert json.loads(st.path.read_text())["model"] == "org/m"


def test_failed_plan_does_not_claim_the_directory(tmp_path):
    RunState.load_or_create(tmp_path / "run", model="errado/modelo")
    # Sem estado gravado, o mesmo diretório aceita outro modelo em seguida.
    st = RunState.load_or_create(tmp_path / "run", model="certo/modelo")
    st.begin("fetch")
    assert json.loads(st.path.read_text())["model"] == "certo/modelo"


def test_round_trips(tmp_path):
    st = RunState.load_or_create(tmp_path / "r", model="org/m")
    st.begin("fetch")
    st.finish("fetch", outputs={"dir": tmp_path}, metrics={"bytes": 42.0})

    again = RunState.load_or_create(tmp_path / "r")
    assert again.stage("fetch").status is StageStatus.OK
    assert again.stage("fetch").metrics["bytes"] == 42.0


def test_running_stage_is_reset_on_load(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    st.begin("prune")
    assert st.stage("prune").status is StageStatus.RUNNING

    again = RunState.load_or_create(tmp_path / "r")
    assert again.stage("prune").status is StageStatus.PENDING


def test_is_done_requires_outputs_to_exist(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    out = tmp_path / "artefato"
    out.mkdir()
    st.finish("prune", outputs={"dir": out})
    assert st.is_done("prune", require=("dir",))

    out.rmdir()
    assert not st.is_done("prune", require=("dir",))


def test_is_done_false_for_pending(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    assert not st.is_done("fetch")


def test_skip_counts_as_done(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    st.skip("export", "desligado")
    assert st.is_done("export")
    assert st.stage("export").status is StageStatus.SKIPPED


def test_fail_records_error(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    st.begin("recover")
    st.fail("recover", "sem memória")
    assert not st.is_done("recover")
    assert RunState.load_or_create(tmp_path / "r").stage("recover").error == "sem memória"


def test_save_is_atomic(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    st.begin("fetch")
    assert not st.path.with_suffix(".tmp").exists()
    json.loads(st.path.read_text())


def test_summary_lists_stages_in_order(tmp_path):
    st = RunState.load_or_create(tmp_path / "r")
    st.finish("fetch")
    st.finish("prune")
    assert [n for n, _, _ in st.summary()] == ["fetch", "prune"]

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from aguardente import compress
from aguardente.evaluation import QualityPolicy
from aguardente.errors import AguardenteError


@pytest.fixture
def search_case(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"immutable checkpoint")
    data = tmp_path / "data.jsonl"
    data.write_text("sample data")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# explicit local adapter")
    spec = dict(model=str(model), data=str(data), adapter=f"{adapter}:build", out=str(tmp_path / "run"),
                task="custom", seq_len=8, seed=42, policy=asdict(QualityPolicy()),
                recipes=["w4-block32", "w8", "none"], max_asset_bytes=None, timeout=30)
    calls = []
    behavior = {"w4-block32": "rejected", "w8": "validated", "none": "validated", "test": "accepted"}

    def run(args, **kwargs):
        final = args[-1] == "test"
        recipe = args[-2] if final else args[-1]
        calls.append((recipe, final))
        folder = Path(spec["out"]) / recipe
        folder.mkdir(exist_ok=True)
        if behavior.get(recipe) == "error":
            return SimpleNamespace(returncode=1)
        row = ({"status": behavior["test"]} if final else
               {"recipe": recipe, "status": behavior.get(recipe, "reference"), "failures": [],
                "bytes": {"w4-block32": 4, "w8": 8, "none": 16}.get(recipe, 0),
                "native_fidelity": {"relative_rmse": 0.01}})
        row["runtime"] = {"call_seconds": [0.1, 0.2], "host_peak_rss_bytes": 1024}
        (folder / ("test.json" if final else "result.json")).write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compress.subprocess, "run", run)
    return spec, calls, behavior


def test_selects_smallest_validated_and_only_tests_winner(search_case):
    spec, calls, _ = search_case
    result = compress.search(spec, report=lambda _: None)
    assert result["selected"] == "w8"
    assert [recipe for recipe, final in calls if final] == ["w8"]
    calls.clear()
    compress.search(spec, report=lambda _: None)
    assert calls == [("w8", True)]


def test_failed_recipe_is_recorded_and_never_selected(search_case):
    spec, calls, behavior = search_case
    behavior["w4-block32"] = "error"
    result = compress.search(spec, report=lambda _: None)
    assert result["selected"] == "w8"
    assert result["candidates"][0]["status"] == "failed"


def test_test_failure_cannot_select_another_candidate(search_case):
    spec, calls, behavior = search_case
    behavior["test"] = "rejected"
    with pytest.raises(AguardenteError, match="teste final"):
        compress.search(spec, report=lambda _: None)
    assert [recipe for recipe, final in calls if final] == ["w8"]
    assert json.loads((Path(spec["out"]) / "selection.json").read_text())["status"] == "rejected"


def test_tampered_reference_report_stops_resume(search_case):
    spec, _, _ = search_case
    compress.search(spec, report=lambda _: None)
    path = Path(spec["out"]) / "baseline/result.json"
    path.write_text("{}")
    with pytest.raises(AguardenteError, match="relatório"):
        compress.search(spec, report=lambda _: None)
    assert json.loads((Path(spec["out"]) / "selection.json").read_text())["status"] == "failed"


def test_any_asset_file_change_invalidates_identity(tmp_path):
    asset = tmp_path / "model.aimodel"
    asset.mkdir()
    weights = asset / "opaque-shard-without-extension"
    weights.write_bytes(b"original")
    before = compress._artifact_identity(asset)
    weights.write_bytes(b"modified")
    assert before != compress._artifact_identity(asset)


def test_latency_objective_and_pareto_tradeoffs():
    def candidate(name, size, latency, error):
        return dict(recipe=name, bytes=size, native_fidelity={"relative_rmse": error},
                    runtime={"call_seconds": [latency], "host_peak_rss_bytes": 100})
    small = candidate("small", 10, 2, 0.02)
    fast = candidate("fast", 20, 1, 0.01)
    dominated = candidate("worse", 30, 3, 0.03)
    assert compress.pareto_candidates([small, fast, dominated]) == [small, fast]

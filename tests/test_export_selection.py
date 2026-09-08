import json
from pathlib import Path

import pytest

from aguardente import export, pipeline
from aguardente.errors import AguardenteError
from tests.test_pipeline_guards import contexto


def fake_export(monkeypatch, ctx, *, runtime_error=False):
    calls = []
    monkeypatch.setattr(export, "build_command", lambda model, out, **kw: [str(out)])

    def run(command, **kwargs):
        out = Path(command[0]) / "bundle"
        out.mkdir(parents=True)
        (out / "metadata.json").write_text("{}")
        asset = out / "model.aimodel"
        asset.mkdir()
        (asset / "weight.bin").write_bytes(b"weights")
        return 0

    monkeypatch.setattr(export, "run_export", run)

    def validate(ctx, model, asset, compression, report, **kwargs):
        calls.append(compression)
        failed = compression == "4bit"
        data = {"passed": not failed, "failures": ["numerical error"] if failed else []}
        if failed and runtime_error:
            data["failure_kind"] = "runtime_error"
        report.write_text(json.dumps(data))
        return int(failed), data

    monkeypatch.setattr(pipeline, "_runtime_result", validate)
    return calls


def test_auto_selects_first_candidate_that_preserves_quality(tmp_path, monkeypatch):
    ctx = contexto(tmp_path, compression="auto")
    calls = fake_export(monkeypatch, ctx)
    result = pipeline.stage_export(ctx, tmp_path / "model")
    assert calls == ["4bit", "8bit"]
    assert ctx.state.stage("export").outputs["compression"] == "8bit"
    assert "8bit" in str(result)
    assert ctx.state.stage("export").status == "ok"
    # Changing recovery settings must not reuse this bundle.
    ctx.opts.lr *= 2
    assert not pipeline._resume(ctx, "export")


def test_runtime_crash_is_not_hidden_by_trying_another_recipe(tmp_path, monkeypatch):
    ctx = contexto(tmp_path, compression="auto")
    calls = fake_export(monkeypatch, ctx, runtime_error=True)
    with pytest.raises(AguardenteError, match="validação"):
        pipeline.stage_export(ctx, tmp_path / "model")
    assert calls == ["4bit"]
    assert ctx.state.stage("export").status != "ok"


def test_explicit_recipe_is_never_silently_replaced(tmp_path, monkeypatch):
    ctx = contexto(tmp_path, compression="4bit")
    calls = fake_export(monkeypatch, ctx)
    with pytest.raises(AguardenteError):
        pipeline.stage_export(ctx, tmp_path / "model")
    assert calls == ["4bit"]


def test_ambiguous_bundle_is_rejected_instead_of_using_latest(tmp_path):
    for name in ("old", "new"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "metadata.json").write_text("{}")
    with pytest.raises(AguardenteError, match="mais de um bundle"):
        export.find_bundle(tmp_path)


def test_embedded_aimodel_metadata_does_not_shadow_bundle_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text('{"language": {"max_context_length": 512}}')
    asset = tmp_path / "model.aimodel"
    asset.mkdir()
    (asset / "metadata.json").write_text('{"model_description": "internal metadata"}')
    result = export.find_bundle(tmp_path)
    assert result.aimodel == asset
    assert result.metadata["language"]["max_context_length"] == 512


def test_multiple_assets_under_same_metadata_are_ambiguous(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "one.aimodel").mkdir()
    (tmp_path / "two.aimodel").mkdir()
    with pytest.raises(AguardenteError, match="mais de um .aimodel"):
        export.find_bundle(tmp_path)


def test_export_timeout_also_covers_silent_processes():
    import subprocess
    import sys
    with pytest.raises(subprocess.TimeoutExpired):
        export.run_export([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1)

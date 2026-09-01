"""Testes de montagem de comandos e exportação para Core AI."""

import pytest

from aguardente.errors import AguardenteError
from aguardente.export import build_command, find_bundle


@pytest.fixture(autouse=True)
def exporter_present(monkeypatch):
    import aguardente.export as mod
    monkeypatch.setattr(mod, "_resolve", lambda _: ["/usr/bin/coreai.llm.export"])


def cmd(**kw):
    return build_command("/tmp/student", "/tmp/out", **kw)


def test_experimental_is_on_by_default():
    assert "--experimental" in cmd()


def test_compute_precision_always_present():
    c = cmd()
    assert "--compute-precision" in c
    assert c[c.index("--compute-precision") + 1] == "float16"


def test_compression_and_config_are_mutually_exclusive():
    preset = cmd()
    assert "--compression" in preset and "--compression-config" not in preset

    yaml = cmd(compression_config="/tmp/recipe.yaml")
    assert "--compression-config" in yaml and "--compression" not in yaml


def test_platform_and_context_are_forwarded():
    c = cmd(platform="iOS", max_context_length=4096)
    assert c[c.index("--platform") + 1] == "iOS"
    assert c[c.index("--max-context-length") + 1] == "4096"


def test_default_compression_is_the_macos_preset():
    c = cmd()
    assert c[c.index("--compression") + 1] == "4bit"


def test_flags_absent_when_not_requested():
    c = cmd()
    for flag in ("--overwrite", "--dry-run", "--include-debug-info"):
        assert flag not in c


def test_missing_bundle_raises_with_path(tmp_path):
    with pytest.raises(AguardenteError) as e:
        find_bundle(tmp_path)
    assert str(tmp_path) in str(e.value)


def test_find_bundle_reads_metadata(tmp_path):
    bundle = tmp_path / "Student-4bit"
    (bundle / "m.aimodel").mkdir(parents=True)
    (bundle / "m.aimodel" / "weights.bin").write_bytes(b"x" * 1024)
    (bundle / "metadata.json").write_text('{"kind": "llm", "compression": "4bit"}')

    result = find_bundle(tmp_path)
    assert result.metadata["kind"] == "llm"
    assert result.aimodel.name == "m.aimodel"
    assert result.size_bytes == 1024


def test_available_reflects_resolution(monkeypatch):
    import aguardente.export as mod

    monkeypatch.setattr(mod, "_resolve", lambda _: None)
    assert not mod.available()

    monkeypatch.setattr(mod, "_resolve", lambda _: ["/usr/bin/coreai.llm.export"])
    assert mod.available()


def test_missing_exporter_error_warns_about_pypi_squat(monkeypatch):
    import aguardente.export as mod

    monkeypatch.setattr(mod, "_resolve", lambda _: None)
    with pytest.raises(AguardenteError) as e:
        mod.build_command("/tmp/m", "/tmp/o")
    assert "PyPI" in (e.value.hint or "")

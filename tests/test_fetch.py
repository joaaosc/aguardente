"""Selecao de arquivos e deteccao de download incompleto."""

import pytest

from aguardente.errors import AguardenteError
from aguardente.fetch import FetchPlan, RemoteFile, _wanted


@pytest.mark.parametrize("path", [
    "config.json", "model.safetensors", "model-00001-of-00003.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "vocab.json", "merges.txt", "generation_config.json",
    "special_tokens_map.json", "chat_template.jinja",
])
def test_wanted_files(path):
    assert _wanted(path), f"{path} deveria ser baixado"


@pytest.mark.parametrize("path", [
    "README.md", "LICENSE", ".gitattributes", "pytorch_model.bin",
    "onnx/model.onnx", "coreml/model.mlpackage", "gguf/model-q4.gguf",
    "openvino/openvino_model.xml",
])
def test_unwanted_files(path):
    """Baixar .bin quando ha .safetensors dobra o trafego a troco de nada."""
    assert not _wanted(path), f"{path} nao deveria ser baixado"


def test_onnx_safetensors_still_excluded():
    """Um .safetensors dentro de onnx/ nao e o peso que queremos."""
    assert not _wanted("onnx/model.safetensors")


def plan_with(tmp_path, files):
    return FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                     files=tuple(RemoteFile(p, s) for p, s in files))


def test_missing_detects_absent_file(tmp_path):
    plan = plan_with(tmp_path, [("config.json", 100)])
    assert len(plan.missing()) == 1
    assert plan.pending_bytes == 100


def test_missing_detects_truncated_file(tmp_path):
    """Um download morto no meio deixa o arquivo com tamanho errado."""
    (tmp_path / "model.safetensors").write_bytes(b"x" * 50)
    plan = plan_with(tmp_path, [("model.safetensors", 100)])
    assert len(plan.missing()) == 1


def test_complete_file_is_not_missing(tmp_path):
    (tmp_path / "config.json").write_bytes(b"x" * 100)
    plan = plan_with(tmp_path, [("config.json", 100)])
    assert plan.missing() == ()
    assert plan.pending_bytes == 0


def test_nested_paths_are_handled(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.safetensors").write_bytes(b"x" * 10)
    plan = plan_with(tmp_path, [("sub/a.safetensors", 10)])
    assert plan.missing() == ()


def test_total_bytes():
    plan = FetchPlan("org/m", "main", None,
                     (RemoteFile("a", 100), RemoteFile("b", 250)))
    assert plan.total_bytes == 350


def test_url_construction():
    f = RemoteFile("model.safetensors", 1)
    assert f.url("Qwen/Qwen3-4B") == \
        "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model.safetensors"


def test_require_aria2_error_names_the_fix(monkeypatch):
    import aguardente.fetch as mod
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    with pytest.raises(AguardenteError) as e:
        mod.require_aria2()
    assert "brew install aria2" in (e.value.hint or "")

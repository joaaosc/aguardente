"""Testes de seleção de arquivos e download via aria2c."""

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
    assert not _wanted(path), f"{path} nao deveria ser baixado"


def test_onnx_safetensors_still_excluded():
    assert not _wanted("onnx/model.safetensors")


def plan_with(tmp_path, files):
    return FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                     files=tuple(RemoteFile(p, s) for p, s in files))


def test_missing_detects_absent_file(tmp_path):
    plan = plan_with(tmp_path, [("config.json", 100)])
    assert len(plan.missing()) == 1
    assert plan.pending_bytes == 100


def test_missing_detects_truncated_file(tmp_path):
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


# --------------------------------------------------- mensagem de erro do 401
#
# O endpoint de listagem de arquivos devolve 200 mesmo para um repositório
# gated — o estado de acesso só entra na hora de baixar o conteúdo de um
# arquivo, não na listagem. Um 401 aqui só pode significar identificador
# inexistente, nunca "gated".


def test_401_na_listagem_e_nao_encontrado(monkeypatch):
    import urllib.error

    import aguardente.fetch as m

    def levanta(*a, **k):
        raise urllib.error.HTTPError("url", 401, "erro", {}, None)

    monkeypatch.setattr(m.urllib.request, "urlopen", levanta)
    with pytest.raises(AguardenteError) as e:
        m.list_files("org/nao-existe")

    assert "não encontrado" in e.value.message
    assert "gated" not in e.value.message and "licença" not in (e.value.hint or "")


def test_arquivo_vazio_sem_tamanho_remoto_conta_como_incompleto(tmp_path):
    """Sem tamanho no índice, a verificação era pulada e um vazio passava."""
    plano = FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                      files=(RemoteFile(path="config.json", size=0),))
    (tmp_path / "config.json").write_bytes(b"")

    assert [f.path for f in plano.missing()] == ["config.json"]


def test_arquivo_com_conteudo_sem_tamanho_remoto_conta_como_completo(tmp_path):
    """Sem referência de tamanho, só o vazio é afirmável como incompleto."""
    plano = FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                      files=(RemoteFile(path="config.json", size=0),))
    (tmp_path / "config.json").write_text("{}")

    assert plano.missing() == ()


def test_tamanho_divergente_conta_como_incompleto(tmp_path):
    plano = FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                      files=(RemoteFile(path="model.safetensors", size=100),))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 40)

    assert [f.path for f in plano.missing()] == ["model.safetensors"]
    assert plano.pending_bytes == 100


def test_checksum_detects_corruption_with_same_size(tmp_path):
    import hashlib
    good = b"original weights"
    file = RemoteFile("model.safetensors", len(good), hashlib.sha256(good).hexdigest())
    plan = FetchPlan("org/m", "main", tmp_path, (file,))
    (tmp_path / file.path).write_bytes(b"X" * len(good))
    assert plan.missing() == (file,)
    (tmp_path / file.path).write_bytes(good)
    assert not plan.missing()


def test_model_tree_follows_pagination_without_dropping_weight_shards(monkeypatch):
    import io
    import json
    from aguardente import fetch as module
    base = "https://huggingface.co/api/models/org/model/tree/main?recursive=1"

    class Response(io.BytesIO):
        def __init__(self, entries, link=""):
            super().__init__(json.dumps(entries).encode())
            self.headers = {"Link": link}

    responses = iter([
        Response([{"type": "file", "path": "config.json", "size": 2}], f'<{base}&cursor=2>; rel="next"'),
        Response([{"type": "file", "path": "model.safetensors", "size": 128}]),
    ])
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return next(responses)

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    assert [f.path for f in module.list_files("org/model")] == ["config.json", "model.safetensors"]
    assert requested == [base, base + "&cursor=2"]


def test_model_tree_rejects_pagination_to_a_different_repository(monkeypatch):
    import io
    from aguardente import fetch as module

    class Response(io.BytesIO):
        headers = {"Link": '<https://huggingface.co/api/models/other/model/tree/main>; rel="next"'}

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: Response(b"[]"))
    with pytest.raises(AguardenteError, match="origem ou revisão"):
        module.list_files("org/model")


def test_download_pins_revision_and_resume_reuses_the_commit(tmp_path, monkeypatch):
    import io
    import json
    from aguardente import fetch as module
    commit = "a" * 40
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps({"sha": commit}).encode()))
    revisions = []
    monkeypatch.setattr(module, "list_files", lambda model, rev: revisions.append(rev) or ())
    monkeypatch.setattr(module, "require_aria2", lambda: "aria2c")
    plan = module.plan_fetch("org/model", tmp_path)
    module.fetch(plan)
    assert plan.revision == commit
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("must reuse pinned revision"))
    assert module.plan_fetch("org/model", tmp_path).revision == commit
    assert revisions == [commit]  # persisted file manifest permits offline resume


def test_tokenizer_and_processor_assets_are_downloaded():
    from aguardente.fetch import _wanted
    for name in ("vocab.txt", "spiece.model", "processor_config.json", "preprocessor_config.json"):
        assert _wanted(name)


def test_old_selection_manifest_refreshes_files_at_original_commit(tmp_path, monkeypatch):
    import json
    from aguardente import fetch as module
    commit = "b" * 40
    (tmp_path / ".aguardente-source.json").write_text(json.dumps({
        "model": "org/model", "requested_revision": "main", "commit": commit, "files": []}))
    revisions = []
    monkeypatch.setattr(module, "list_files", lambda model, rev: revisions.append(rev) or (RemoteFile("vocab.txt", 100),))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("must keep pinned commit"))
    plan = module.plan_fetch("org/model", tmp_path)
    assert revisions == [commit]
    assert plan.files[0].path == "vocab.txt"

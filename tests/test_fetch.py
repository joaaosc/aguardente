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

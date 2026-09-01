"""Defesas contra dados hostis vindos da rede.

O indice de arquivos de um repositorio remoto e controlado por quem o publica.
Tratar esses caminhos como confiaveis permite escrita fora do destino.
"""

import pytest

from aguardente.errors import AguardenteError
from aguardente.fetch import RemoteFile, _wanted, safe_join

# Cada entrada e um caminho que um repositorio malicioso poderia declarar.
TRAVESSIAS = [
    "../../../../etc/cron.d/pwned.safetensors",
    "../../.ssh/authorized_keys.safetensors",
    "/etc/passwd.safetensors",
    "subdir/../../fora.safetensors",
    "..\\..\\windows.safetensors",
    "./../escape.safetensors",
]

LEGITIMOS = [
    "config.json",
    "model.safetensors",
    "model-00001-of-00003.safetensors",
    "tokenizer.json",
]


@pytest.mark.parametrize("path", TRAVESSIAS)
def test_wanted_rejeita_travessia(path):
    """O filtro de selecao e a primeira linha de defesa."""
    assert not _wanted(path), f"{path} passou pelo filtro"


@pytest.mark.parametrize("path", LEGITIMOS)
def test_wanted_aceita_legitimos(path):
    assert _wanted(path), f"{path} foi barrado indevidamente"


@pytest.mark.parametrize("path", TRAVESSIAS)
def test_safe_join_rejeita_travessia(tmp_path, path):
    """Defesa em profundidade: mesmo que o filtro falhe, a juncao recusa."""
    with pytest.raises(AguardenteError):
        safe_join(tmp_path, path)


def test_safe_join_aceita_subdiretorio(tmp_path):
    destino = safe_join(tmp_path, "sub/model.safetensors")
    assert destino == tmp_path / "sub" / "model.safetensors"
    assert str(destino).startswith(str(tmp_path))


def test_safe_join_normaliza_sem_escapar(tmp_path):
    """Um './' redundante e inofensivo e deve passar normalizado."""
    assert safe_join(tmp_path, "./config.json") == tmp_path / "config.json"


def test_remote_file_valida_na_construcao():
    """Construir um RemoteFile com caminho hostil deve falhar de imediato."""
    with pytest.raises(AguardenteError):
        RemoteFile(path="../fora.safetensors", size=10)


def test_remote_file_aceita_legitimo():
    f = RemoteFile(path="model.safetensors", size=10)
    assert f.path == "model.safetensors"


def test_url_nao_permite_injecao_de_host():
    """O caminho nao pode redirecionar o download para outro servidor."""
    with pytest.raises(AguardenteError):
        RemoteFile(path="https://malicioso.example/x.safetensors", size=1)


# ------------------------------------------------ identificador de modelo

IDS_HOSTIS = [
    "../../api/internal",
    "a/b?x=1#frag",
    "a/../../etc",
    "modelo?token=roubado",
    "http://outro.example/x",
    "espaco no meio",
    "a/b/c/d",          # profundidade inválida no formato do Hugging Face
    "",
]

IDS_VALIDOS = [
    "Qwen/Qwen3-4B",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "gpt2",
    "org-com-hifen/modelo.v2",
]


@pytest.mark.parametrize("model_id", IDS_HOSTIS)
def test_identificador_hostil_recusado(model_id):
    """Um identificador com '..', '?' ou '#' reescreve a URL da API."""
    from aguardente.fetch import validate_model_id
    with pytest.raises(AguardenteError):
        validate_model_id(model_id)


@pytest.mark.parametrize("model_id", IDS_VALIDOS)
def test_identificador_valido_aceito(model_id):
    from aguardente.fetch import validate_model_id
    assert validate_model_id(model_id) == model_id


# ------------------------------------------------ desserializacao

def test_logits_carregam_sem_executar_pickle(tmp_path):
    """torch.load sem weights_only executa codigo arbitrario ao desserializar."""
    torch = pytest.importorskip("torch")
    import inspect

    from aguardente.distill import teacher

    fonte = inspect.getsource(teacher.TeacherLogits.batches)
    assert "weights_only=True" in fonte, \
        "torch.load precisa restringir a desserializacao a tensores"


def test_shard_com_objeto_arbitrario_e_recusado(tmp_path):
    """Prova pratica: um shard com objeto de classe arbitraria nao carrega.

    `weights_only=True` permite tipos de dado (tensores, dicts, strings) e
    recusa a reconstrucao de classes — que e por onde um pickle hostil executa
    codigo. `argparse.Namespace` serve de sentinela por ser inofensivo e estar
    fora da lista permitida.
    """
    torch = pytest.importorskip("torch")
    import argparse
    import json as _json

    from aguardente.distill.teacher import TeacherLogits

    (tmp_path / "manifest.json").write_text(
        _json.dumps({"top_k": 4, "shards": 1, "samples": 1, "seq_len": 2}))
    torch.save({"payload": argparse.Namespace(cmd="rm -rf /")}, tmp_path / "000000.pt")

    logits = TeacherLogits.load(tmp_path)
    with pytest.raises(Exception, match="(?i)unpickl|weights_only|global"):
        next(iter(logits.batches()))


def test_shard_legitimo_continua_carregando(tmp_path):
    """A defesa nao pode quebrar o caminho normal."""
    torch = pytest.importorskip("torch")
    import json as _json

    from aguardente.distill.teacher import TeacherLogits

    (tmp_path / "manifest.json").write_text(
        _json.dumps({"top_k": 4, "shards": 1, "samples": 1, "seq_len": 2}))
    torch.save({"input_ids": torch.zeros(1, 2, dtype=torch.long),
                "values": torch.zeros(1, 2, 4),
                "indices": torch.zeros(1, 2, 4, dtype=torch.int32)},
               tmp_path / "000000.pt")

    blob = next(iter(TeacherLogits.load(tmp_path).batches()))
    assert blob["values"].shape == (1, 2, 4)


# ------------------------------------------------ robustez dos parametros

@pytest.mark.parametrize("kwargs", [
    {"connections": 0},
    {"connections": 999},
    {"connections": -1},
    {"concurrent": 0},
    {"max_tries": 10_000},
    {"retry_wait": -5},
    {"connections": "8"},        # string onde se espera inteiro
])
def test_parametros_absurdos_recusados(tmp_path, kwargs):
    """Valores fora de faixa fazem o aria2 falhar com mensagem obscura."""
    from aguardente.fetch import FetchPlan, RemoteFile, fetch

    plan = FetchPlan(model_id="org/m", revision="main", dest=tmp_path,
                     files=(RemoteFile("model.safetensors", 10),))
    with pytest.raises(AguardenteError, match="(?i)precisa ser"):
        fetch(plan, **kwargs)


def test_checkpoint_e_gravado_atomicamente(tmp_path):
    """Escrever direto sobre best.pt destroi o unico checkpoint bom se cair no meio."""
    import inspect

    from aguardente.distill import train

    fonte = inspect.getsource(train._save_checkpoint)
    assert ".replace(" in fonte, "o checkpoint precisa ser renomeado, nao escrito direto"
    assert "tmp" in fonte


def test_historico_de_perdas_tem_teto():
    """Um treino de horas acumularia centenas de milhares de floats sem uso."""
    from aguardente.distill.train import _MAX_LOSS_HISTORY
    assert 0 < _MAX_LOSS_HISTORY <= 10_000

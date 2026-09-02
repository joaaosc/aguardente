"""Resolução de URLs coladas do navegador para o identificador do modelo.

`_hf_exists` e `_hf_search` são as únicas funções que tocam rede; os testes
substituem as duas por monkeypatch, então nenhum caso aqui depende de conexão.
"""

import urllib.error

import pytest

from aguardente.errors import AguardenteError
from aguardente.source import _hf_exists, resolve_source


class _RespostaHTTP:
    """Simula o retorno de `urllib.request.urlopen` sem tocar rede."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _levanta(codigo):
    def _abrir(*a, **k):
        raise urllib.error.HTTPError("url", codigo, "erro", {}, None)
    return _abrir


# ------------------------------------------ mapeamento de status HTTP real
#
# A API do Hugging Face devolve 401 — não 404 — para um identificador que não
# existe, e 200 (não 401) para um repositório gated de verdade: o estado de
# acesso só muda o código de status no download dos arquivos, não aqui. Uma
# implementação ingênua ("401 = gated, logo existe") erra ao contrário do que
# parece: ela reportaria QUALQUER identificador inventado como confirmado.


def test_200_e_existencia(monkeypatch):
    import aguardente.source as m
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: _RespostaHTTP())
    assert _hf_exists("Qwen/Qwen3-4B") is True


@pytest.mark.parametrize("codigo", [401, 404])
def test_401_e_404_sao_ausencia(monkeypatch, codigo):
    """401 é o que a API devolve hoje para repositório inexistente — não gated."""
    import aguardente.source as m
    monkeypatch.setattr(m.urllib.request, "urlopen", _levanta(codigo))
    assert _hf_exists("org/nao-existe") is False


def test_outro_codigo_http_nao_vira_ausencia_silenciosa(monkeypatch):
    """Um 500, por exemplo, não deve ser lido como 'não existe'."""
    import aguardente.source as m
    monkeypatch.setattr(m.urllib.request, "urlopen", _levanta(500))
    with pytest.raises(AguardenteError):
        _hf_exists("org/repo")


def sempre_existe(_id):
    return True


def nunca_existe(_id):
    return False


def sem_candidatos(_valor):
    return []


# --------------------------------------------------------------- passagem direta


def test_diretorio_local_passa_direto(tmp_path):
    d = tmp_path / "modelo"
    d.mkdir()
    assert resolve_source(str(d)) == str(d)


def test_identificador_bare_passa_direto():
    assert resolve_source("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_url_de_host_desconhecido_passa_direto():
    """Não é HF nem GitHub: a validação de identificador cuida disso adiante."""
    assert resolve_source("https://example.com/org/repo") == "https://example.com/org/repo"


def test_string_vazia_nao_quebra():
    assert resolve_source("") == ""


# --------------------------------------------------------------- URL do Hugging Face


@pytest.mark.parametrize("url,esperado", [
    ("https://huggingface.co/Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
    ("https://huggingface.co/Qwen/Qwen3-4B/", "Qwen/Qwen3-4B"),
    ("https://huggingface.co/Qwen/Qwen3-4B/tree/main", "Qwen/Qwen3-4B"),
    ("https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json", "Qwen/Qwen3-4B"),
    ("http://huggingface.co/Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
    ("https://www.huggingface.co/Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
])
def test_url_hf_normalizada(url, esperado):
    assert resolve_source(url) == esperado


def test_url_hf_nao_consulta_a_api(monkeypatch):
    """A URL do HF já é o identificador: não há palpite a confirmar."""
    import aguardente.source as m

    def explode(*a, **k):
        raise AssertionError("não deveria consultar a API para uma URL do HF")

    monkeypatch.setattr(m, "_hf_exists", explode)
    monkeypatch.setattr(m, "_hf_by_author", explode)
    monkeypatch.setattr(m, "_hf_by_name", explode)
    assert resolve_source("https://huggingface.co/Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_url_hf_relata_a_normalizacao():
    avisos = []
    resolve_source("https://huggingface.co/Qwen/Qwen3-4B", report=avisos.append)
    assert avisos and "Qwen/Qwen3-4B" in avisos[0]


# ----------------------------------------------------------------- URL do GitHub


def test_url_github_confirmada_no_hf(monkeypatch):
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", sempre_existe)
    monkeypatch.setattr(m, "_hf_by_author", sem_candidatos)
    assert resolve_source("https://github.com/Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_url_github_com_git_no_final(monkeypatch):
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", sempre_existe)
    monkeypatch.setattr(m, "_hf_by_author", sem_candidatos)
    assert resolve_source("https://github.com/Qwen/Qwen3-4B.git") == "Qwen/Qwen3-4B"


def test_url_github_relata_a_origem_e_o_destino(monkeypatch):
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", sempre_existe)
    monkeypatch.setattr(m, "_hf_by_author", sem_candidatos)
    avisos = []
    resolve_source("https://github.com/Qwen/Qwen3-4B", report=avisos.append)
    assert any("GitHub" in a and "Qwen/Qwen3-4B" in a for a in avisos)


def test_url_github_lista_apenas_repos_do_mesmo_autor(monkeypatch):
    """A advertência é rotulada "do mesmo autor" e só pode conter repos que
    a consulta por autor devolveu — nunca resultado de busca por nome, que
    pode vir de qualquer organização."""
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", sempre_existe)
    monkeypatch.setattr(m, "_hf_by_author", lambda autor: ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"])
    monkeypatch.setattr(m, "_hf_by_name",
                        lambda termo: (_ for _ in ()).throw(
                            AssertionError("não deveria buscar por nome quando já existe")))
    avisos = []
    resolve_source("https://github.com/Qwen/Qwen3-4B", report=avisos.append)
    assert any("Qwen/Qwen3-8B" in a and "mesmo autor" in a for a in avisos)
    # a própria candidata resolvida não deve reaparecer na lista
    assert not any("mesmo autor" in a and "Qwen3-4B," in a for a in avisos)


def test_url_github_sem_analogo_e_recusada_com_candidatos_por_autor(monkeypatch):
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", nunca_existe)
    monkeypatch.setattr(m, "_hf_by_author", lambda autor: ["outra-org/Modelo-7B", "outra-org/Modelo-13B"])
    monkeypatch.setattr(m, "_hf_by_name", sem_candidatos)
    with pytest.raises(AguardenteError) as e:
        resolve_source("https://github.com/outra-org/Modelo")

    assert "outra-org/Modelo" in e.value.message
    assert "do autor" in e.value.message
    assert "outra-org/Modelo-7B" in e.value.message
    assert "outra-org/Modelo-13B" in e.value.message


def test_url_github_candidatos_por_nome_nao_viram_mesmo_autor(monkeypatch):
    """Repositórios achados só por busca de nome, de outro dono, ganham rótulo
    próprio — nunca aparecem como "do autor <owner>", que seria falso."""
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", nunca_existe)
    monkeypatch.setattr(m, "_hf_by_author", sem_candidatos)
    monkeypatch.setattr(m, "_hf_by_name", lambda termo: ["outra-empresa/pytorch-utils"])
    with pytest.raises(AguardenteError) as e:
        resolve_source("https://github.com/pytorch/pytorch")

    assert "outra-empresa/pytorch-utils" in e.value.message
    assert "nome parecido" in e.value.message
    assert "do autor 'pytorch'" not in e.value.message


def test_url_github_sem_analogo_e_sem_candidato_nenhum(monkeypatch):
    import aguardente.source as m

    monkeypatch.setattr(m, "_hf_exists", nunca_existe)
    monkeypatch.setattr(m, "_hf_by_author", sem_candidatos)
    monkeypatch.setattr(m, "_hf_by_name", sem_candidatos)
    with pytest.raises(AguardenteError) as e:
        resolve_source("https://github.com/alguem/projeto-qualquer")

    assert "não encontrei" in e.value.message
    assert "GitHub" in e.value.hint


def test_url_github_nunca_adivinha_alem_da_busca(monkeypatch):
    """A recusa não tenta variações do nome — só usa o que a API devolveu."""
    import aguardente.source as m

    chamadas = []

    monkeypatch.setattr(m, "_hf_exists", nunca_existe)
    monkeypatch.setattr(m, "_hf_by_author", lambda autor: chamadas.append(("autor", autor)) or [])
    monkeypatch.setattr(m, "_hf_by_name", lambda termo: chamadas.append(("nome", termo)) or [])
    with pytest.raises(AguardenteError):
        resolve_source("https://github.com/deepseek-ai/DeepSeek-VL")

    assert ("autor", "deepseek-ai") in chamadas
    assert ("nome", "DeepSeek-VL") in chamadas
    assert len(chamadas) == 2


def test_url_github_propaga_falha_de_rede(monkeypatch):
    import aguardente.source as m

    def rede_fora(_id):
        raise AguardenteError("falha de rede ao consultar x no Hugging Face")

    monkeypatch.setattr(m, "_hf_exists", rede_fora)
    with pytest.raises(AguardenteError) as e:
        resolve_source("https://github.com/org/repo")

    assert "falha de rede" in e.value.message

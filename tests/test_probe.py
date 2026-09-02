"""Testes de `probe._get_json`: tradução de falha HTTP em erro acionável.

O mesmo helper serve tanto para os metadados de um modelo quanto para o
config.json do repositório, e um 401 nos dois cobre duas causas distintas —
identificador inexistente, ou repositório gated — sem como diferenciá-las
pelo código de status sozinho. A mensagem precisa admitir as duas, nunca
escolher uma.
"""

import urllib.error

import pytest

from aguardente.errors import ProbeError
from aguardente.probe import _get_json


def _levanta(codigo):
    def _abrir(*a, **k):
        raise urllib.error.HTTPError("url", codigo, "erro", {}, None)
    return _abrir


def test_401_admite_inexistente_ou_gated(monkeypatch):
    import aguardente.probe as m

    monkeypatch.setattr(m.urllib.request, "urlopen", _levanta(401))
    with pytest.raises(ProbeError) as e:
        _get_json("https://huggingface.co/api/models/org/repo")

    # Não pode afirmar que é gated — a API também devolve 401 para um
    # identificador que simplesmente não existe.
    assert "gated" not in e.value.message
    assert "Se o repositório existir e for gated" in (e.value.hint or "")


def test_404_e_nao_encontrado(monkeypatch):
    import aguardente.probe as m

    monkeypatch.setattr(m.urllib.request, "urlopen", _levanta(404))
    with pytest.raises(ProbeError) as e:
        _get_json("https://huggingface.co/api/models/org/repo")

    assert "não encontrado" in e.value.message

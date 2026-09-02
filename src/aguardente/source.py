"""Resolução de URLs coladas do navegador para o identificador do modelo.

O usuário frequentemente copia a URL da página do Hugging Face — ou, por
hábito, do GitHub, onde muitos autores hospedam o código do modelo sem os
pesos. Este módulo normaliza os dois casos para o identificador
`namespace/nome` que o resto do pacote já entende.

A URL do Hugging Face é só sintaxe: o identificador já está nela, e a
normalização é uma conferência mecânica. A URL do GitHub é uma aposta — o
nome do repositório às vezes coincide com o do Hugging Face, às vezes não —
e por isso nunca é aceita em silêncio: só depois de confirmada contra a API.
Quando a aposta falha, os candidatos mais próximos são mostrados, e a escolha
fica com quem executou o comando.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from .errors import AguardenteError

_HF = "https://huggingface.co"
_TIMEOUT = 15
_MAX_CANDIDATES = 8

_URL = re.compile(r"^https?://(?:www\.)?(?P<host>[a-z0-9.-]+)/(?P<owner>[^/?#]+)/(?P<repo>[^/?#]+)")

Reporter = Callable[[str], None]


def _sem_aviso(_msg: str) -> None:
    return None


def _parse_repo_url(ref: str) -> tuple[str, str, str] | None:
    """(host, owner, repo) para uma URL de huggingface.co ou github.com; senão None."""
    m = _URL.match(ref.strip())
    if not m:
        return None
    host = m["host"].lower()
    if host not in ("huggingface.co", "github.com"):
        return None
    return host, m["owner"], m["repo"].removesuffix(".git")


def _hf_exists(model_id: str) -> bool:
    """Confere a existência do repositório sem baixar nada.

    A API devolve 200 para qualquer repositório existente, gated ou não — o
    estado de gated não muda o código de status neste endpoint, só no
    download dos arquivos (`resolve/main/...`). Um identificador que não
    existe devolve 401, não 404 — conferido contra vários repositórios
    fantasma antes de escrever este código, porque a suposição óbvia (401 =
    gated, 404 = ausente) está invertida na API atual.
    """
    try:
        urllib.request.urlopen(f"{_HF}/api/models/{model_id}", timeout=_TIMEOUT)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            return False
        raise AguardenteError(f"HTTP {e.code} ao consultar {model_id} no Hugging Face") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise AguardenteError(f"falha de rede ao consultar {model_id} no Hugging Face: {e}") from e


def _hf_query(param: str, valor: str) -> list[str]:
    """Uma única consulta à API de listagem de modelos, tolerante a falha de rede.

    Erro aqui não deve derrubar a resolução: sem candidatos, a mensagem de
    erro fica mais pobre, mas ainda é honesta — nunca inventa um nome que a
    API não devolveu.
    """
    try:
        url = f"{_HF}/api/models?{param}={urllib.parse.quote(valor)}&limit={_MAX_CANDIDATES}"
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            return [mid for item in json.loads(r.read()) if (mid := item.get("id"))]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return []


def _hf_by_author(autor: str) -> list[str]:
    """Repositórios publicados por este autor — o sentido literal de "mesmo autor"."""
    return _hf_query("author", autor)[:_MAX_CANDIDATES]


def _hf_by_name(termo: str) -> list[str]:
    """Repositórios cujo nome bate com o termo, de qualquer autor.

    Não é filtrado por autor porque a API de busca por texto já não o é: os
    resultados podem vir de qualquer organização, e apresentá-los como "do
    mesmo autor" seria falso.
    """
    return _hf_query("search", termo)[:_MAX_CANDIDATES]


def resolve_source(ref: str, *, report: Reporter = _sem_aviso) -> str:
    """Normaliza uma URL do Hugging Face ou do GitHub para `namespace/nome`.

    Um diretório local passa direto — o mesmo teste que o resto do pacote já
    usa para distinguir referência local de identificador remoto. Uma URL do
    GitHub só é aceita depois de confirmada: quando o nome do repositório não
    tem análogo no Hugging Face, a função levanta um erro listando os
    candidatos mais próximos, em vez de arriscar um palpite.
    """
    if Path(ref).expanduser().exists():
        return ref

    achado = _parse_repo_url(ref)
    if achado is None:
        return ref
    host, owner, repo = achado
    candidato = f"{owner}/{repo}"

    if host == "huggingface.co":
        report(f"origem       {ref} → {candidato}")
        return candidato

    # github.com: o nome do repositório é só um palpite de identificador HF,
    # e só é aceito depois de confirmado contra a API. A confirmação é o único
    # motivo de existir — não há adivinhação de nome nem correspondência
    # aproximada: ou "owner/repo" existe tal como está, ou a função para.
    if _hf_exists(candidato):
        report(f"origem       {ref} (GitHub) → {candidato} (Hugging Face)")
        irmaos = [m for m in _hf_by_author(owner) if m != candidato]
        if irmaos:
            report(f"             outros modelos do mesmo autor no Hugging Face: "
                   f"{', '.join(irmaos[:5])}")
        return candidato

    por_autor = [m for m in _hf_by_author(owner) if m != candidato]
    por_nome = [m for m in _hf_by_name(repo) if m != candidato and m not in por_autor]
    if not por_autor and not por_nome:
        raise AguardenteError(
            f"não encontrei {candidato!r} no Hugging Face",
            hint="O GitHub costuma hospedar só o código do modelo, não os pesos. "
                 f"Procure o repositório correspondente em https://huggingface.co/{owner} "
                 "e informe o identificador (namespace/nome) diretamente.",
        )
    partes = [f"{candidato!r} não existe no Hugging Face."]
    if por_autor:
        partes.append("repositórios do autor " + repr(owner) + ":\n    "
                      + "\n    ".join(por_autor))
    if por_nome:
        partes.append("repositórios com nome parecido, de outros autores:\n    "
                      + "\n    ".join(por_nome))
    raise AguardenteError(
        "\n  ".join(partes),
        hint="Rode novamente com um dos identificadores acima, no formato namespace/nome.",
    )

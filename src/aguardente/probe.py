"""Inspeção de metadados e estrutura do modelo antes do download completo."""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arch import Arch, count_stored
from .errors import ProbeError

_HF = "https://huggingface.co"
_TIMEOUT = 30
_MAX_HEADER = 64 * 1024 * 1024
_MAX_JSON = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelProbe:
    """Metadados e estrutura identificados na sondagem do modelo."""

    ref: str
    arch: Arch
    model_type: str
    stored_params: int
    is_local: bool = False
    gated: bool = False
    pipeline_tag: str | None = None
    lm_head_materialized: bool = False
    weight_files: tuple[str, ...] = ()
    tensor_names: tuple[str, ...] = field(default=(), repr=False)

    @property
    def name(self) -> str:
        return Path(self.ref).name if self.is_local else self.ref

    @property
    def fp16_bytes(self) -> int:
        return self.stored_params * 2

    def count_error(self) -> float:
        """Erro relativo entre o cálculo analítico e o total real de parâmetros."""
        counted = count_stored(self.arch, lm_head_materialized=self.lm_head_materialized)
        return (counted - self.stored_params) / self.stored_params


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            bruto = r.read(_MAX_JSON + 1)
            if len(bruto) > _MAX_JSON:
                raise ProbeError(f"resposta de {url} excede o limite máximo permitido")
            return json.loads(bruto)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ProbeError(
                f"acesso negado a {url}",
                hint="Modelo gated. Aceite a licença na página do Hugging Face e rode `hf auth login`.",
            ) from e
        if e.code == 404:
            raise ProbeError(f"não encontrado: {url}", hint="Confira o identificador informado.") from e
        raise ProbeError(f"HTTP {e.code} em {url}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ProbeError(f"falha de rede em {url}: {e}") from e


def read_safetensors_header(url_or_path: str) -> dict[str, Any]:
    """Lê o cabeçalho JSON de um arquivo .safetensors (local ou remoto via Range)."""
    if not url_or_path.startswith("http"):
        with open(url_or_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > _MAX_HEADER:
                raise ProbeError(f"tamanho de cabeçalho inválido ({n} bytes) em {url_or_path}")
            return json.loads(f.read(n))

    def _range(a: int, b: int) -> bytes:
        req = urllib.request.Request(url_or_path, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.read()

    try:
        n = struct.unpack("<Q", _range(0, 7))[0]
        if n > _MAX_HEADER:
            raise ProbeError(f"tamanho de cabeçalho inválido ({n} bytes)")
        return json.loads(_range(8, 8 + n - 1))
    except (urllib.error.URLError, TimeoutError, struct.error, json.JSONDecodeError) as e:
        raise ProbeError(f"não foi possível ler o cabeçalho de {url_or_path}: {e}") from e


def _summarize_tensors(header: dict[str, Any]) -> tuple[int, bool, tuple[str, ...]]:
    """Extrai contagem de parâmetros, presença de lm_head e nomes dos tensores do cabeçalho."""
    header = {k: v for k, v in header.items() if k != "__metadata__"}
    total = 0
    for meta in header.values():
        n = 1
        for d in meta.get("shape", []):
            n *= d
        total += n
    has_lm_head = any("lm_head" in k for k in header)
    return total, has_lm_head, tuple(sorted(header))


def probe_local(path: str | Path) -> ModelProbe:
    """Inspeciona os metadados de um modelo em diretório local."""
    d = Path(path).expanduser().resolve()
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        raise ProbeError(
            f"{d} não contém config.json",
            hint="O diretório precisa conter a estrutura padrão: config.json, arquivos .safetensors e tokenizer.",
        )
    cfg = json.loads(cfg_path.read_text())
    arch = Arch.from_hf_config(cfg)

    shards = sorted(p for p in d.glob("*.safetensors"))
    if not shards:
        raise ProbeError(f"{d} não contém arquivos .safetensors")

    total, has_lm_head, names = 0, False, []
    for s in shards:
        t, h, n = _summarize_tensors(read_safetensors_header(str(s)))
        total += t
        has_lm_head |= h
        names.extend(n)

    return ModelProbe(
        ref=str(d),
        arch=arch,
        model_type=str(cfg.get("model_type", "")),
        stored_params=total,
        is_local=True,
        lm_head_materialized=has_lm_head,
        weight_files=tuple(s.name for s in shards),
        tensor_names=tuple(sorted(names)),
    )


def probe_hub(model_id: str) -> ModelProbe:
    """Inspeciona os metadados de um modelo no Hugging Face Hub sem baixar pesos."""
    from .fetch import validate_model_id

    validate_model_id(model_id)
    api = _get_json(f"{_HF}/api/models/{model_id}")
    cfg = _get_json(f"{_HF}/{model_id}/resolve/main/config.json")
    arch = Arch.from_hf_config(cfg)

    files = tuple(
        s["rfilename"] for s in api.get("siblings", [])
        if s["rfilename"].endswith(".safetensors")
    )
    if not files:
        raise ProbeError(
            f"{model_id} não publica pesos em .safetensors",
            hint="Só o formato safetensors é suportado.",
        )

    reported = (api.get("safetensors") or {}).get("total")

    has_lm_head = False
    names: tuple[str, ...] = ()
    try:
        hdr = read_safetensors_header(f"{_HF}/{model_id}/resolve/main/{files[0]}")
        shard_total, has_lm_head, names = _summarize_tensors(hdr)
        if reported is None:
            reported = shard_total if len(files) == 1 else None
    except ProbeError:
        pass

    if reported is None:
        raise ProbeError(
            f"não foi possível determinar a contagem de parâmetros de {model_id}",
            hint="O modelo pode ser multi-shard sem metadados na API.",
        )

    return ModelProbe(
        ref=model_id,
        arch=arch,
        model_type=str(cfg.get("model_type", "")),
        stored_params=int(reported),
        gated=bool(api.get("gated")),
        pipeline_tag=api.get("pipeline_tag"),
        lm_head_materialized=has_lm_head,
        weight_files=files,
        tensor_names=names,
    )


def probe(ref: str) -> ModelProbe:
    """Inspeciona o modelo a partir de um identificador do Hub ou diretório local."""
    p = Path(ref).expanduser()
    if p.exists() and p.is_dir():
        return probe_local(p)
    return probe_hub(ref)

"""Inspeção de metadados e estrutura do modelo antes do download completo."""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arch import Arch, count_stored, text_config
from .errors import ProbeError
from .textonly import TextLayout, arch_from_shapes, discover_layout

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
    # Presente quando o checkpoint foi inspecionado tensor a tensor. Ausente
    # quando só houve metadados da API, caso em que texto e visão não se separam.
    layout: TextLayout | None = field(default=None, repr=False)
    text_model_type: str = ""

    @property
    def name(self) -> str:
        return Path(self.ref).name if self.is_local else self.ref

    @property
    def text_params(self) -> int:
        """Parâmetros do decoder de texto, que é o que o pipeline processa."""
        return self.layout.kept_params if self.layout else self.stored_params

    @property
    def dropped_params(self) -> int:
        """Parâmetros de torre de visão, projetor e demais componentes descartados."""
        return self.layout.dropped_params if self.layout else 0

    @property
    def is_multimodal(self) -> bool:
        return bool(self.layout and self.layout.is_multimodal)

    @property
    def fp16_bytes(self) -> int:
        return self.text_params * 2

    def count_error(self) -> float:
        """Erro relativo entre o cálculo analítico e o total real de parâmetros.

        A comparação é contra o decoder, não contra o checkpoint inteiro: num
        modelo multimodal a torre de visão responde por parte dos tensores e
        nunca entra na contagem analítica.
        """
        counted = count_stored(self.arch, lm_head_materialized=self.lm_head_materialized)
        return (counted - self.text_params) / self.text_params


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            bruto = r.read(_MAX_JSON + 1)
            if len(bruto) > _MAX_JSON:
                raise ProbeError(f"resposta de {url} excede o limite máximo permitido")
            return json.loads(bruto)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            # A API devolve 401 tanto para um identificador que não existe
            # quanto para um repositório gated sem autenticação — o mesmo
            # código de status cobre os dois casos, e não há como
            # distingui-los aqui sem uma segunda chamada.
            raise ProbeError(
                f"acesso negado a {url}",
                hint="Confira o identificador. Se o repositório existir e for "
                     "gated, baixe o modelo autorizado com o cliente HF e use "
                     "o diretório local.",
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


# Ler o cabeçalho de cada shard custa duas requisições por arquivo. Acima deste
# teto o custo deixa de compensar, e a sondagem cai para os metadados da API —
# que dão o total, mas não separam texto de visão.
MAX_SHARDS_SONDADOS = 32


def _totalizar(headers: dict[str, dict[str, Any]]) -> tuple[int, bool, list[str]]:
    """Soma os parâmetros de todos os shards e reúne os nomes."""
    total, has_lm_head, names = 0, False, []
    for header in headers.values():
        t, h, n = _summarize_tensors(header)
        total += t
        has_lm_head |= h
        names.extend(n)
    return total, has_lm_head, names


def _analisar(cfg: dict[str, Any],
              headers: dict[str, dict[str, Any]]) -> tuple[TextLayout | None, Arch, str]:
    """Localiza o decoder e deriva as dimensões das formas dos tensores.

    O config entra apenas onde as formas são ambíguas. Quando não há cabeçalhos
    — sondagem que só alcançou os metadados da API —, resta o config sozinho.
    """
    sub, _ = text_config(cfg)
    architectures = sub.get("architectures", cfg.get("architectures", []))
    if architectures and all(name.endswith("Model") or "Classification" in name
                             for name in architectures):
        raise ProbeError(
            f"checkpoint não causal: {architectures}",
            hint="Use um modelo treinado para geração (ForCausalLM). "
                 "Um encoder de embeddings não fornece logits causais treinados.",
        )
    tipo = str(sub.get("model_type") or cfg.get("model_type") or "")
    if not headers:
        return None, Arch.from_hf_config(cfg), tipo
    limpos = {a: {k: v for k, v in h.items() if k != "__metadata__"}
              for a, h in headers.items()}
    layout = discover_layout(limpos)
    return layout, arch_from_shapes(limpos, layout, cfg=sub), tipo


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

    shards = sorted(p for p in d.glob("*.safetensors"))
    if not shards:
        raise ProbeError(f"{d} não contém arquivos .safetensors")

    headers = {s.name: read_safetensors_header(str(s)) for s in shards}
    total, has_lm_head, names = _totalizar(headers)
    layout, arch, tipo_texto = _analisar(cfg, headers)

    return ModelProbe(
        ref=str(d),
        arch=arch,
        model_type=str(cfg.get("model_type", "")),
        stored_params=total,
        is_local=True,
        lm_head_materialized=has_lm_head,
        weight_files=tuple(s.name for s in shards),
        tensor_names=tuple(sorted(names)),
        layout=layout,
        text_model_type=tipo_texto,
    )


def probe_hub(model_id: str) -> ModelProbe:
    """Inspeciona os metadados de um modelo no Hugging Face Hub sem baixar pesos."""
    from .fetch import validate_model_id

    validate_model_id(model_id)
    api = _get_json(f"{_HF}/api/models/{model_id}")
    cfg = _get_json(f"{_HF}/{model_id}/resolve/main/config.json")

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

    # Separar decoder de torre de visão exige as formas de todos os tensores, e
    # não só as do primeiro shard: o decoder do DeepSeek-VL se espalha por três.
    headers: dict[str, dict[str, Any]] = {}
    if len(files) <= MAX_SHARDS_SONDADOS:
        try:
            for nome in files:
                headers[nome] = read_safetensors_header(
                    f"{_HF}/{model_id}/resolve/main/{nome}")
        except ProbeError:
            headers = {}

    has_lm_head, names = False, ()
    if headers:
        shard_total, has_lm_head, nomes = _totalizar(headers)
        names = tuple(sorted(nomes))
        if reported is None:
            reported = shard_total

    if reported is None:
        raise ProbeError(
            f"não foi possível determinar a contagem de parâmetros de {model_id}",
            hint="O modelo pode ser multi-shard sem metadados na API.",
        )

    layout, arch, tipo_texto = _analisar(cfg, headers)

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
        layout=layout,
        text_model_type=tipo_texto,
    )


def probe(ref: str) -> ModelProbe:
    """Inspeciona o modelo a partir de um identificador do Hub ou diretório local."""
    p = Path(ref).expanduser()
    if p.exists() and p.is_dir():
        return probe_local(p)
    return probe_hub(ref)

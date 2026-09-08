"""Geração e leitura de shards de logits do modelo teacher."""

from __future__ import annotations

import json
import hashlib
import math
import pickle
import warnings
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from ..errors import AguardenteError

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_TOP_K = 128
_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class TeacherLogits:
    """Gerenciador de shards de logits pré-computados."""

    path: Path
    top_k: int
    shards: int
    samples: int
    seq_len: int
    temperature: float = 2.0
    tail_samples: int = 0

    @classmethod
    def load(cls, path: str | Path) -> TeacherLogits:
        p = Path(path)
        meta = json.loads((p / _MANIFEST).read_text())
        return cls(path=p, top_k=meta["top_k"], shards=meta["shards"],
                   samples=meta["samples"], seq_len=meta["seq_len"],
                   temperature=meta.get("temperature", 2.0), tail_samples=meta.get("tail_samples", 0))

    def batches(self, *, device: str | None = None,
                start: int = 0, order: list[int] | None = None) -> Iterator[dict[str, "torch.Tensor"]]:
        """Itera sobre os shards armazenados em ordem, a partir de `start`.

        Retomar pulando com `islice` custaria uma leitura de disco e uma
        transferência para o dispositivo por shard descartado; começar o
        `range` adiante não lê nada do que já foi treinado.
        """
        import torch

        order = list(range(self.shards)) if order is None else order
        if sorted(order) != list(range(self.shards)):
            raise ValueError("shard order must be a permutation")
        for i in order[max(0, start):]:
            blob = torch.load(self.path / f"{i:06d}.pt", map_location="cpu",
                              weights_only=True)
            if device:
                blob = {k: v.to(device) for k, v in blob.items()}
            yield blob

    def estimated_bytes(self) -> int:
        """Valores FP32, índices int32 e normalizador FP32 por token."""
        return self.samples * self.seq_len * ((self.top_k + self.tail_samples) * 8 + 4)


def estimate_logit_bytes(samples: int, seq_len: int, *, top_k: int = DEFAULT_TOP_K,
                         tail_samples: int = 0,
                         vocab_size: int | None = None) -> tuple[int, int | None]:
    """Estima o espaço em disco necessário para armazenamento dos logits."""
    topk_bytes = samples * seq_len * ((top_k + tail_samples) * 8 + 4)
    full_bytes = samples * seq_len * vocab_size * 2 if vocab_size else None
    return topk_bytes, full_bytes


def precompute_logits(
    teacher: Any,
    batches: Iterable[dict[str, "torch.Tensor"]],
    out_dir: str | Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 2.0,
    max_batches: int | None = None,
    on_progress: Any = None,
    cache_key: str | None = None,
    tail_samples: int = 0,
    seed: int = 42,
) -> TeacherLogits:
    """Executa o modelo teacher e grava os top-k logits em shards por lote."""
    import torch

    if temperature <= 0 or not math.isfinite(temperature) or top_k < 1 or tail_samples < 0:
        raise ValueError("temperature must be positive")

    if cache_key is None and hasattr(teacher, "state_dict"):
        digest = hashlib.sha256()
        for name, tensor in teacher.state_dict().items():
            digest.update(name.encode())
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        cache_key = digest.hexdigest()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # A assinatura em cada shard permite retomar uma geração interrompida antes
    # do manifesto, sem misturar teacher, dados ou parâmetros de outra execução.
    reaproveita = cache_key is not None
    signature = hashlib.sha256(json.dumps([cache_key, top_k, temperature, tail_samples, seed]).encode()).digest()
    cache_tag = torch.tensor(list(signature), dtype=torch.uint8)
    manifesto_antigo = out / _MANIFEST
    if manifesto_antigo.is_file():
        try:
            old = json.loads(manifesto_antigo.read_text())
            gravado = old.get("top_k")
            reaproveita = (old.get("format_version") == 3 and old.get("temperature") == temperature
                           and cache_key is not None and old.get("cache_key") == cache_key)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            gravado = None
        if gravado is not None and int(gravado) != int(top_k):
            reaproveita = False

    was_training = teacher.training
    teacher.eval()
    shards = samples = seq_len = valid_tokens = 0
    mass_sum = mass_tokens = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(islice(batches, max_batches)):
                shard_path = out / f"{i:06d}.pt"
                input_ids = batch["input_ids"]
                mask = batch.get("attention_mask", torch.ones_like(input_ids))
                valid_tokens += int((mask[:, :-1].bool() & mask[:, 1:].bool()).sum())
                if reaproveita and shard_path.is_file() and shard_path.stat().st_size > 0:
                    try:
                        saved = torch.load(shard_path, weights_only=True, map_location="cpu")
                    except (OSError, RuntimeError, EOFError, IndexError, ValueError, pickle.UnpicklingError) as error:
                        warnings.warn(f"shard ilegível {shard_path.name}; regenerando com o teacher: {error}", stacklevel=2)
                        saved = {}
                    if ("logsumexp" in saved and "cache_tag" in saved and torch.equal(saved["cache_tag"], cache_tag)
                            and torch.equal(saved["input_ids"], input_ids.cpu())
                            and torch.equal(saved.get("attention_mask", torch.ones_like(saved["input_ids"])), mask.cpu())
                            and math.isclose(saved["temperature"].item(), temperature, rel_tol=1e-6)):
                        top_mass = (saved["values"] / temperature - saved["logsumexp"].unsqueeze(-1)).exp().sum(-1)
                        active = mask.cpu().bool()
                        mass_sum += float(top_mass[active].sum())
                        mass_tokens += int(active.sum())
                        shards += 1
                        samples += int(input_ids.size(0))
                        seq_len = int(input_ids.size(1))
                        if on_progress:
                            on_progress(shards)
                        continue

                logits = teacher(**batch).logits
                k = min(top_k, logits.size(-1))
                values, indices = logits.topk(k, dim=-1)
                # A cauda também participa do normalizador. -Inf fora do top-k
                # não apareceria numa verificação restrita aos valores salvos.
                if not torch.isfinite(logits).all():
                    raise AguardenteError(
                        "o modelo teacher produziu valores não finitos (NaN ou Inf) durante a geração de logits",
                        hint="Verifique se os pesos do modelo original ou o dispositivo estão corrompendo os tensores.",
                    )

                payload = {
                    "cache_tag": cache_tag,
                    "input_ids": input_ids.detach().cpu(),
                    "values": values.detach().float().cpu(),
                    "logsumexp": torch.logsumexp(logits.float() / temperature, dim=-1).cpu(),
                    "temperature": torch.tensor(temperature),
                    "indices": indices.detach().to(torch.int32).cpu(),
                }
                if "attention_mask" in batch:
                    payload["attention_mask"] = batch["attention_mask"].detach().to(torch.int8).cpu()
                top_mass = (values.float().cpu() / temperature - payload["logsumexp"].unsqueeze(-1)).exp().sum(-1)
                active = mask.cpu().bool()
                mass_sum += float(top_mass[active].sum())
                mass_tokens += int(active.sum())
                if tail_samples and k < logits.size(-1):
                    # Sample on CPU with an explicit generator: reproducible on
                    # MPS/CPU/CUDA and independent of training/dropout RNG.
                    probs = (logits.detach().float().cpu() / temperature - payload["logsumexp"].unsqueeze(-1)).exp()
                    probs.scatter_(-1, indices.cpu(), 0)
                    flat = probs.reshape(-1, probs.size(-1))
                    zero = flat.sum(-1) == 0
                    flat[zero, 0] = 1  # zero-mass tails have zero weight in KD
                    generator = torch.Generator().manual_seed(seed + i)
                    sampled = torch.multinomial(flat, tail_samples, replacement=True, generator=generator)
                    sampled = sampled.reshape(*input_ids.shape, tail_samples)
                    payload["tail_indices"] = sampled.to(torch.int32)
                    payload["tail_values"] = logits.detach().float().cpu().gather(-1, sampled)

                tmp_shard = shard_path.with_suffix(".pt.tmp")
                try:
                    torch.save(payload, tmp_shard)
                    tmp_shard.replace(shard_path)
                finally:
                    tmp_shard.unlink(missing_ok=True)

                shards += 1
                samples += int(input_ids.size(0))
                seq_len = int(input_ids.size(1))
                if on_progress:
                    on_progress(shards)
    finally:
        teacher.train(was_training)

    if shards == 0:
        raise AguardenteError("nenhum lote de logits foi produzido")
    meta = {"top_k": top_k, "shards": shards, "samples": samples, "seq_len": seq_len,
            "temperature": temperature, "tail_samples": tail_samples}
    tmp_manifest = out / f"{_MANIFEST}.tmp"
    try:
        tmp_manifest.write_text(json.dumps({**meta, "format_version": 3, "cache_key": cache_key,
                                            "valid_tokens": valid_tokens,
                                            "topk_probability_mass": mass_sum / mass_tokens if mass_tokens else None}, indent=2))
        tmp_manifest.replace(out / _MANIFEST)
    finally:
        tmp_manifest.unlink(missing_ok=True)
    return TeacherLogits(path=out, **meta)

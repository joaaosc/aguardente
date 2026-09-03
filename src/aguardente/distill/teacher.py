"""Geração e leitura de shards de logits do modelo teacher."""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    @classmethod
    def load(cls, path: str | Path) -> TeacherLogits:
        p = Path(path)
        meta = json.loads((p / _MANIFEST).read_text())
        return cls(path=p, top_k=meta["top_k"], shards=meta["shards"],
                   samples=meta["samples"], seq_len=meta["seq_len"])

    def batches(self, *, device: str | None = None) -> Iterator[dict[str, "torch.Tensor"]]:
        """Itera sobre os shards armazenados em ordem."""
        import torch

        for i in range(self.shards):
            blob = torch.load(self.path / f"{i:06d}.pt", map_location="cpu",
                              weights_only=True)
            if device:
                blob = {k: v.to(device) for k, v in blob.items()}
            yield blob

    def estimated_bytes(self) -> int:
        """Estimativa de bytes para valores fp16 e índices int32 por token."""
        return self.samples * self.seq_len * self.top_k * (2 + 4)


def estimate_logit_bytes(samples: int, seq_len: int, *, top_k: int = DEFAULT_TOP_K,
                         vocab_size: int | None = None) -> tuple[int, int | None]:
    """Estima o espaço em disco necessário para armazenamento dos logits."""
    topk_bytes = samples * seq_len * top_k * (2 + 4)
    full_bytes = samples * seq_len * vocab_size * 2 if vocab_size else None
    return topk_bytes, full_bytes


def precompute_logits(
    teacher: Any,
    batches: Iterable[dict[str, "torch.Tensor"]],
    out_dir: str | Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    on_progress: Any = None,
) -> TeacherLogits:
    """Executa o modelo teacher e grava os top-k logits em shards por lote."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    was_training = teacher.training
    teacher.eval()
    shards = samples = seq_len = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(batches):
                shard_path = out / f"{i:06d}.pt"
                input_ids = batch["input_ids"]
                if shard_path.exists():
                    shards += 1
                    samples += int(input_ids.size(0))
                    seq_len = int(input_ids.size(1))
                    continue

                logits = teacher(**batch).logits
                if not torch.isfinite(logits).all():
                    raise AguardenteError(
                        "o modelo teacher produziu valores não finitos (NaN ou Inf) durante a geração de logits",
                        hint="Verifique se os pesos do modelo original ou o dispositivo estão corrompendo os tensores.",
                    )
                k = min(top_k, logits.size(-1))
                values, indices = logits.topk(k, dim=-1)

                payload = {
                    "input_ids": input_ids.detach().cpu(),
                    "values": values.detach().half().cpu(),
                    "indices": indices.detach().to(torch.int32).cpu(),
                }
                if "attention_mask" in batch:
                    payload["attention_mask"] = batch["attention_mask"].detach().to(torch.int8).cpu()

                torch.save(payload, shard_path)
                shards += 1
                samples += int(input_ids.size(0))
                seq_len = int(input_ids.size(1))
                if on_progress:
                    on_progress(shards)
    finally:
        teacher.train(was_training)

    meta = {"top_k": top_k, "shards": shards, "samples": samples, "seq_len": seq_len}
    (out / _MANIFEST).write_text(json.dumps(meta, indent=2))
    return TeacherLogits(path=out, **meta)

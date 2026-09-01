"""Pré-computação dos logits do teacher.

O teacher só faz forward, sempre sob `no_grad`. Mantê-lo residente durante
várias épocas paga a memória dele N vezes pelos mesmos números — e em máquina
pequena essa memória é justamente o que falta.

Fazendo o forward uma vez e guardando o top-k em disco, o pico de RAM passa a
ser `max(teacher, student)` em vez de `teacher + student`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_TOP_K = 128
_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class TeacherLogits:
    """Um diretório de shards com o top-k do teacher."""

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
        """Itera os shards gravados, na ordem."""
        import torch

        for i in range(self.shards):
            # weights_only=True recusa pickles arbitrários: `torch.load` sem
            # esta restrição executa código durante a desserialização, e estes
            # shards podem vir de um backup ou de outra máquina.
            blob = torch.load(self.path / f"{i:06d}.pt", map_location="cpu",
                              weights_only=True)
            if device:
                blob = {k: v.to(device) for k, v in blob.items()}
            yield blob

    def estimated_bytes(self) -> int:
        """valores fp16 + índices int32, por token."""
        return self.samples * self.seq_len * self.top_k * (2 + 4)


def estimate_logit_bytes(samples: int, seq_len: int, *, top_k: int = DEFAULT_TOP_K,
                         vocab_size: int | None = None) -> tuple[int, int | None]:
    """Bytes com top-k e, para comparação, com o vocabulário completo.

    A segunda grandeza é o que torna óbvio por que o top-k não é opcional.
    """
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
    """Roda o teacher e grava o top-k por lote.

    Grava um shard por lote — retomar uma corrida interrompida é só pular os
    shards que já existem.
    """
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
                k = min(top_k, logits.size(-1))
                values, indices = logits.topk(k, dim=-1)

                torch.save(
                    {
                        "input_ids": input_ids.detach().cpu(),
                        "values": values.detach().half().cpu(),
                        "indices": indices.detach().to(torch.int32).cpu(),
                    },
                    shard_path,
                )
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

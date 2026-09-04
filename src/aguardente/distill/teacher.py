"""Geração e leitura de shards de logits do modelo teacher."""

from __future__ import annotations

import json
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

    @classmethod
    def load(cls, path: str | Path) -> TeacherLogits:
        p = Path(path)
        meta = json.loads((p / _MANIFEST).read_text())
        return cls(path=p, top_k=meta["top_k"], shards=meta["shards"],
                   samples=meta["samples"], seq_len=meta["seq_len"])

    def batches(self, *, device: str | None = None,
                start: int = 0) -> Iterator[dict[str, "torch.Tensor"]]:
        """Itera sobre os shards armazenados em ordem, a partir de `start`.

        Retomar pulando com `islice` custaria uma leitura de disco e uma
        transferência para o dispositivo por shard descartado; começar o
        `range` adiante não lê nada do que já foi treinado.
        """
        import torch

        for i in range(max(0, start), self.shards):
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
    max_batches: int | None = None,
    on_progress: Any = None,
) -> TeacherLogits:
    """Executa o modelo teacher e grava os top-k logits em shards por lote."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Shards de uma execução anterior só podem ser reaproveitados se tiverem sido
    # gerados com o mesmo top-k. O manifesto é a única fonte confiável disso: sem
    # essa checagem, mudar `--top-k` reaproveitaria silenciosamente logits com K
    # antigo enquanto o manifesto passaria a anunciar o K novo.
    reaproveita = True
    manifesto_antigo = out / _MANIFEST
    if manifesto_antigo.is_file():
        try:
            gravado = json.loads(manifesto_antigo.read_text()).get("top_k")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            gravado = None
        if gravado is not None and int(gravado) != int(top_k):
            reaproveita = False

    was_training = teacher.training
    teacher.eval()
    shards = samples = seq_len = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(islice(batches, max_batches)):
                shard_path = out / f"{i:06d}.pt"
                input_ids = batch["input_ids"]
                if reaproveita and shard_path.is_file() and shard_path.stat().st_size > 0:
                    shards += 1
                    samples += int(input_ids.size(0))
                    seq_len = int(input_ids.size(1))
                    if on_progress:
                        on_progress(shards)
                    continue

                logits = teacher(**batch).logits
                k = min(top_k, logits.size(-1))
                values, indices = logits.topk(k, dim=-1)
                # `topk` ordena NaN acima de qualquer número finito e preserva ±Inf,
                # então checar os k valores selecionados detecta a mesma corrupção
                # que varrer [B, T, V] inteiro, a uma fração do custo.
                if not torch.isfinite(values).all():
                    raise AguardenteError(
                        "o modelo teacher produziu valores não finitos (NaN ou Inf) durante a geração de logits",
                        hint="Verifique se os pesos do modelo original ou o dispositivo estão corrompendo os tensores.",
                    )

                payload = {
                    "input_ids": input_ids.detach().cpu(),
                    "values": values.detach().half().cpu(),
                    "indices": indices.detach().to(torch.int32).cpu(),
                }
                if "attention_mask" in batch:
                    payload["attention_mask"] = batch["attention_mask"].detach().to(torch.int8).cpu()

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

    meta = {"top_k": top_k, "shards": shards, "samples": samples, "seq_len": seq_len}
    tmp_manifest = out / f"{_MANIFEST}.tmp"
    try:
        tmp_manifest.write_text(json.dumps(meta, indent=2))
        tmp_manifest.replace(out / _MANIFEST)
    finally:
        tmp_manifest.unlink(missing_ok=True)
    return TeacherLogits(path=out, **meta)

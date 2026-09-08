"""Output fidelity and explicit, finite task metrics for any model adapter."""
from dataclasses import dataclass, field
import math
import re
import unicodedata

from .errors import AguardenteError


@dataclass(frozen=True)
class QualityPolicy:
    max_relative_rmse: float = 0.05
    max_loss_ratio: float = 1.05
    max_score_drop: float = 0.02
    min_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("max_relative_rmse", "max_loss_ratio", "max_score_drop"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise AguardenteError(f"{name} precisa ser finito e não negativo")
        if self.max_loss_ratio < 1:
            raise AguardenteError("max_loss_ratio precisa ser >= 1")
        if not isinstance(self.min_scores, dict) or any(
            not isinstance(key, str) or not key.startswith("score:") or not isinstance(value, (int, float)) or not math.isfinite(value)
            for key, value in self.min_scores.items()):
            raise AguardenteError("min_scores exige métricas score: e limites finitos")

    def failures(self, fidelity: dict, baseline: dict, candidate: dict) -> list[str]:
        errors = []
        if not math.isfinite(fidelity["relative_rmse"]) or fidelity["relative_rmse"] > self.max_relative_rmse:
            errors.append("output_relative_rmse")
        if set(candidate) != set(baseline):
            errors.append("task_metric_contract")
            return errors
        for key, value in candidate.items():
            reference = baseline[key]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (reference, value)):
                errors.append(f"nonfinite:{key}")
            elif key.startswith("loss:"):
                if min(reference, value) < 0 or value > reference * self.max_loss_ratio:
                    errors.append(key)
            elif key.startswith("score:"):
                if value < reference - self.max_score_drop:
                    errors.append(key)
            else:
                raise AguardenteError(f"métrica {key!r} deve declarar direção loss: ou score:")
        for key, minimum in self.min_scores.items():
            if key not in candidate or candidate[key] < minimum:
                errors.append(f"absolute:{key}")
        return errors


def output_fidelity(reference, candidate) -> dict:
    import numpy as np
    if len(reference) != len(candidate) or not reference:
        raise AguardenteError("quantidade de amostras de avaliação incompatível ou vazia")
    error = energy = 0.0
    elements = 0
    max_abs = 0.0
    per_output = {}
    for original, changed in zip(reference, candidate, strict=True):
        if len(original) != len(changed) or not original:
            raise AguardenteError("contrato de saídas incompatível")
        for index, (a, b) in enumerate(zip(original, changed, strict=True)):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if a.shape != b.shape or a.size == 0:
                raise AguardenteError("formas de saída incompatíveis ou vazias")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AguardenteError("saídas não finitas durante avaliação")
            diff = b - a
            error += float(np.square(diff).sum())
            energy += float(np.square(a).sum())
            sums = per_output.setdefault(index, [0.0, 0.0])
            sums[0] += float(np.square(diff).sum())
            sums[1] += float(np.square(a).sum())
            elements += a.size
            max_abs = max(max_abs, float(np.abs(diff).max()))
    return {"relative_rmse": max(math.sqrt(err / max(power, 1e-24)) for err, power in per_output.values()),
            "global_relative_rmse": math.sqrt(error / max(energy, 1e-24)),
            "rmse": math.sqrt(error / elements), "max_abs": max_abs, "elements": elements}


def answer_metrics(answer: str, expected: list[str]) -> dict:
    """Exact factual checks require supplied accepted answers, never teacher agreement."""
    def normalize(text):
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
    if not isinstance(answer, str) or not isinstance(expected, list) or not expected or any(not isinstance(x, str) or not normalize(x) for x in expected):
        raise AguardenteError("informe respostas de referência não vazias")
    tokens = normalize(answer).split()
    longest = 0
    for width in range(1, min(8, len(tokens) // 3) + 1):
        run = width
        for i in range(width, len(tokens)):
            run = run + 1 if tokens[i] == tokens[i - width] else width
            if run >= 3 * width:
                longest = max(longest, run)
    return {"score:exact_match": float(normalize(answer) in {normalize(x) for x in expected}),
            "loss:repeated_token_fraction": longest / max(1, len(tokens))}


def cmd_score_answers(args):
    """Score supplied generations against accepted answers, without a model judge."""
    import json
    from pathlib import Path
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (args.min_exact_match, args.max_repetition)):
        raise AguardenteError("limites de respostas precisam estar entre 0 e 1")
    results = []
    for line in Path(args.data).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row.get("answer"), str) or not isinstance(row.get("expected"), list):
                raise AguardenteError("cada linha precisa conter answer textual e expected como lista de respostas aceitas")
            results.append(answer_metrics(row["answer"], row["expected"]))
    if not results:
        raise AguardenteError("arquivo sem respostas")
    metrics = {k: sum(row[k] for row in results) / len(results) for k in results[0]}
    failed = metrics["score:exact_match"] < args.min_exact_match or metrics["loss:repeated_token_fraction"] > args.max_repetition
    print(json.dumps({"status": "rejected" if failed else "accepted", "samples": len(results), "metrics": metrics,
                      "scope": "exact accepted answers and periodic token runs; not a general factuality benchmark"}))
    return 1 if failed else 0

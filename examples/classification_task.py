"""Custom task example for an already trained dense feature classifier.

Checkpoint: config.json with input_features, hidden_features, classes; and
model.safetensors for Sequential(Linear, ReLU, Linear). Data: JSONL records with
features, label and split (calibration/validation/test). No training occurs here.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from aguardente.tasks import ModelTask, SPLITS


def build(model_ref, data_path):
    root = Path(model_ref)
    config = json.loads((root / "config.json").read_text())
    model = torch.nn.Sequential(
        torch.nn.Linear(config["input_features"], config["hidden_features"]),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_features"], config["classes"]),
    )
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    samples = {s: [] for s in SPLITS}
    labels = {s: [] for s in SPLITS}
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tensor = torch.tensor([row["features"]], dtype=torch.float32)
            if tensor.shape != (1, config["input_features"]):
                raise ValueError("features must match the checkpoint input dimension")
            samples[row["split"]].append((tensor,))
            labels[row["split"]].append(row["label"])

    def score(outputs, split):
        logits = torch.cat([torch.as_tensor(row[0]) for row in outputs])
        targets = torch.tensor(labels[split], dtype=torch.long)
        return {"loss:cross_entropy": torch.nn.functional.cross_entropy(logits, targets).item(),
                "score:accuracy": (logits.argmax(-1) == targets).float().mean().item()}

    return ModelTask(model.eval(), ("features",), ("logits",), samples, score,
                     description="dense-feature-classifier",
                     preprocessing={"input_features": config["input_features"],
                                    "note": "features must use the training normalization"}).validate()

import json

import pytest

from aguardente.cli import main


def test_score_answers_rejects_factually_wrong_and_repeated_answers(tmp_path, capsys):
    data = tmp_path / "answers.jsonl"
    data.write_text(json.dumps({"answer": "Venus Venus Venus", "expected": ["Mercury"]}))
    assert main(["score-answers", str(data)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rejected"
    assert result["metrics"]["score:exact_match"] == 0
    assert result["metrics"]["loss:repeated_token_fraction"] == 1


def test_inspect_reports_noncausal_task_without_importing_pruning(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert", "architectures": ["BertForMaskedLM"]}))
    assert main(["inspect", str(tmp_path), "--target-ram-gib", "8"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task"] == "masked-lm"
    assert result["target"]["ram_gib"] == 8


def test_custom_factory_loads_all_trained_weights_and_scores(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from aguardente.tasks import load_factory
    config = dict(input_features=4, hidden_features=8, classes=2)
    (tmp_path / "config.json").write_text(json.dumps(config))
    original = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    save_file(original.state_dict(), str(tmp_path / "model.safetensors"))
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({"features": [i, 0, 1, 2], "label": 0, "split": split})
                  for i, split in enumerate(("calibration", "validation", "test"))))
    task = load_factory("examples/classification_task.py:build")(str(tmp_path), str(path))
    for name, parameter in task.model.state_dict().items():
        assert torch.equal(parameter, original.state_dict()[name])
    logits = task.model(*task.samples["test"][0])
    assert "score:accuracy" in task.score([(logits,)], "test")

import pytest
from aguardente.evaluation import QualityPolicy, answer_metrics, output_fidelity
from aguardente.errors import AguardenteError


def test_truth_is_not_teacher_agreement():
    assert answer_metrics("Vênus", ["Mercúrio"])["score:exact_match"] == 0
    assert answer_metrics("Mercúrio.", ["mercúrio"])["score:exact_match"] == 1
    assert answer_metrics("que que que que", ["outra resposta"])["loss:repeated_token_fraction"] == 1


def test_policy_rejects_nonfinite_missing_and_degraded_metrics():
    policy = QualityPolicy()
    assert policy.failures({"relative_rmse": 0.01}, {"loss:nll": 1}, {"loss:nll": 1.2})
    assert policy.failures({"relative_rmse": 0.01}, {"score:accuracy": 1}, {})
    assert policy.failures({"relative_rmse": 0.01}, {"score:accuracy": 1}, {"score:accuracy": float("nan")})
    with pytest.raises(AguardenteError):
        QualityPolicy(max_relative_rmse=float("inf"))


def test_fidelity_rejects_shape_changes():
    np = pytest.importorskip("numpy")
    with pytest.raises(AguardenteError):
        output_fidelity([(np.zeros((2, 3)),)], [(np.zeros((3, 2)),)])


def test_adapter_split_contamination_is_rejected():
    torch = pytest.importorskip("torch")
    from aguardente.tasks import ModelTask
    data = {s: [(torch.tensor([[1., 2.]]),)] for s in ("calibration", "validation", "test")}
    with pytest.raises(AguardenteError, match="contaminação"):
        ModelTask(torch.nn.Linear(2, 2), ("input",), ("output",), data).validate()


def test_bad_reference_cannot_pass_an_absolute_task_floor():
    policy = QualityPolicy(min_scores={"score:accuracy": 0.8})
    assert "absolute:score:accuracy" in policy.failures({"relative_rmse": 0},
                {"score:accuracy": 0}, {"score:accuracy": 0})


def test_large_output_cannot_hide_corrupted_small_output():
    np = pytest.importorskip("numpy")
    before = [(np.ones(10000), np.ones(1))]
    after = [(np.ones(10000), np.zeros(1))]
    result = output_fidelity(before, after)
    assert result["global_relative_rmse"] < 0.02
    assert result["relative_rmse"] == 1

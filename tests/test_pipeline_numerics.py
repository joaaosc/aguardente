"""Behavioral regressions found by the real-model audit."""
import copy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.test_distill import tiny_model, fixed_batches
from aguardente.distill.loss import kd_loss
from aguardente.distill.teacher import precompute_logits
from aguardente.distill.train import recover, RecoveryConfig
from aguardente.prune.scoring import score_model
from aguardente.prune.surgery import _resize_per_layer_lists


def test_kd_penalizes_probability_outside_teacher_topk():
    teacher = torch.tensor([[[3., 2., 0., -1.]]])
    values, indices = teacher.topk(2, -1)
    logz = torch.logsumexp(teacher / 2, -1)
    good = kd_loss(teacher, values, indices, alpha=1, teacher_logsumexp=logz)
    student = teacher.clone().requires_grad_()
    bad = student + torch.tensor([[[0., 0., 10., 0.]]])
    loss = kd_loss(bad, values, indices, alpha=1, teacher_logsumexp=logz)
    loss.backward()
    assert good.item() == pytest.approx(0, abs=1e-6)
    assert loss.item() > 1
    assert student.grad[0, 0, 2] > 0


def test_single_optimizer_step_really_updates_bfloat16_weights(tmp_path):
    logits = precompute_logits(tiny_model(), fixed_batches(n=1), tmp_path)
    student = tiny_model(seed=2).bfloat16()
    before = {k: v.float().clone() for k, v in student.state_dict().items()}
    result = recover(student, logits, RecoveryConfig(epochs=1, grad_accum=4,
                     gradient_checkpointing=False), device="cpu")
    assert result.steps == 1
    assert next(student.parameters()).dtype == torch.float32
    changed = sum(torch.count_nonzero(v - before[k]).item()
                  for k, v in student.state_dict().items())
    assert changed > sum(v.numel() for v in before.values()) // 2


def test_residual_accumulation_has_same_update_as_full_batch(tmp_path):
    teacher = tiny_model()
    batches = fixed_batches(n=2, bs=1)
    small = precompute_logits(teacher, batches, tmp_path / "small")
    large = precompute_logits(teacher, [{"input_ids": torch.cat([b["input_ids"] for b in batches])}],
                              tmp_path / "large")
    a = tiny_model(seed=2)
    b = copy.deepcopy(a)
    recover(a, small, RecoveryConfig(epochs=1, grad_accum=4, gradient_checkpointing=False))
    recover(b, large, RecoveryConfig(epochs=1, grad_accum=1, gradient_checkpointing=False))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-4, atol=1e-6)


def test_scoring_ignores_right_padding():
    model = tiny_model()
    ids = fixed_batches(n=1, bs=1, seq=8)[0]["input_ids"]
    plain = score_model(model, [{"input_ids": ids}])
    padded = score_model(model, [{"input_ids": torch.cat([ids, torch.ones_like(ids)], 1),
                                  "attention_mask": torch.tensor([[1]*8 + [0]*8])}])
    for name in ("ffn", "kv_groups", "layers"):
        torch.testing.assert_close(getattr(plain, name), getattr(padded, name), atol=1e-6, rtol=1e-4)


def test_layer_lists_do_not_rewrite_unrelated_metadata():
    from types import SimpleNamespace
    config = SimpleNamespace(layer_types=["a", "b", "c"], eos_token_id=[1, 2, 3])
    _resize_per_layer_lists(config, [0, 2], 3)
    assert config.layer_types == ["a", "c"]
    assert config.eos_token_id == [1, 2, 3]

"""Testes de funções de perda e treinamento de recuperação."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

from aguardente.arch import Arch, count_params
from aguardente.distill.loss import kd_loss
from aguardente.distill.teacher import estimate_logit_bytes, precompute_logits
from aguardente.distill.train import RecoveryConfig, recover
from aguardente.plan import PrunePlan
from aguardente.prune.surgery import prune_model
from aguardente.verify import recovery_fraction


def tiny_model(seed=0, **over):
    base = dict(hidden_size=64, intermediate_size=256, num_hidden_layers=4,
                num_attention_heads=8, num_key_value_heads=4, vocab_size=256,
                max_position_embeddings=128, tie_word_embeddings=True)
    base.update(over)
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(**base)).eval()


def fixed_batches(n=6, bs=2, seq=16, vocab=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, vocab, (bs, seq), generator=g)} for _ in range(n)]


def train_briefly(model, batches, *, steps=40, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        for b in batches:
            loss = model(input_ids=b["input_ids"], labels=b["input_ids"]).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    model.eval()
    return model


# ---------------------------------------------------------------- loss

def test_kd_loss_is_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 64)
    vals, idx = logits.topk(8, dim=-1)
    loss = kd_loss(logits, vals, idx, labels=None, alpha=1.0, temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_kd_loss_grows_with_divergence():
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 64)
    vals, idx = teacher.topk(8, dim=-1)
    near = teacher + 0.05 * torch.randn_like(teacher)
    far = teacher + 2.0 * torch.randn_like(teacher)
    a = float(kd_loss(near, vals, idx, alpha=1.0))
    b = float(kd_loss(far, vals, idx, alpha=1.0))
    assert 0 < a < b


def test_temperature_squared_keeps_gradient_scale():
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 64)
    student = (teacher + 0.5 * torch.randn_like(teacher)).requires_grad_(True)
    vals, idx = teacher.topk(8, dim=-1)

    grads = []
    for T in (1.0, 4.0):
        s = student.detach().clone().requires_grad_(True)
        kd_loss(s, vals, idx, alpha=1.0, temperature=T).backward()
        grads.append(float(s.grad.abs().mean()))
    assert 0.1 < grads[1] / grads[0] < 10.0


def test_alpha_blends_both_terms():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 64)
    vals, idx = (logits + torch.randn_like(logits)).topk(8, dim=-1)
    labels = torch.randint(0, 64, (2, 6))
    pure_kd = float(kd_loss(logits, vals, idx, labels, alpha=1.0))
    mixed = float(kd_loss(logits, vals, idx, labels, alpha=0.5))
    assert mixed != pytest.approx(pure_kd, rel=1e-3)


# ---------------------------------------------------------------- teacher

def test_topk_storage_is_orders_of_magnitude_smaller():
    topk, full = estimate_logit_bytes(10_000, 512, top_k=128, vocab_size=151_936)
    assert full / topk > 200
    assert full > 1e12
    assert topk < 5e10


def test_precompute_writes_shards_and_manifest(tmp_path):
    teacher = tiny_model()
    out = precompute_logits(teacher, fixed_batches(n=4), tmp_path / "logits", top_k=8)
    assert out.shards == 4
    assert (tmp_path / "logits" / "manifest.json").is_file()
    blob = next(iter(out.batches()))
    assert blob["values"].shape == (2, 16, 8)
    assert blob["indices"].dtype == torch.int32


def test_precompute_resumes_without_recomputing(tmp_path):
    teacher = tiny_model()
    precompute_logits(teacher, fixed_batches(n=3), tmp_path / "l", top_k=8)
    mtimes = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "l").glob("*.pt")}
    again = precompute_logits(teacher, fixed_batches(n=3), tmp_path / "l", top_k=8)
    assert again.shards == 3
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "l").glob("*.pt")} == mtimes


# ---------------------------------------------------------------- recuperacao

def _mean_kd(model, logits) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for b in logits.batches():
            out = model(input_ids=b["input_ids"])
            total += float(kd_loss(out.logits, b["values"], b["indices"], alpha=1.0))
    return total / logits.shards


def test_pruning_degrades_a_trained_model(tmp_path):
    batches = fixed_batches(n=6)
    teacher = train_briefly(tiny_model(seed=0), batches)
    logits = precompute_logits(teacher, batches, tmp_path / "logits", top_k=16)

    intact = _mean_kd(teacher, logits)
    assert intact == pytest.approx(0.0, abs=1e-4)

    student = train_briefly(tiny_model(seed=0), batches)
    a = Arch.from_hf_config(student.config.to_dict())
    prune_model(student, PrunePlan(source=a, target=a.with_(intermediate_size=128),
                                   requested_params=0),
                keep_ffn=torch.arange(128))
    assert _mean_kd(student, logits) > 0.01


def test_recovery_reduces_loss_on_pruned_student(tmp_path):
    batches = fixed_batches(n=8)
    teacher = train_briefly(tiny_model(seed=0), batches)
    logits = precompute_logits(teacher, batches, tmp_path / "logits", top_k=16)

    student = train_briefly(tiny_model(seed=0), batches)
    a = Arch.from_hf_config(student.config.to_dict())
    prune_model(student, PrunePlan(source=a, target=a.with_(intermediate_size=128),
                                   requested_params=0),
                keep_ffn=torch.arange(128))

    before = _mean_kd(student, logits)
    res = recover(student, logits,
                  RecoveryConfig(epochs=3, learning_rate=1e-3, grad_accum=1,
                                 warmup_fraction=0.1, gradient_checkpointing=False),
                  device="cpu")
    after = _mean_kd(student, logits)

    assert res.steps > 0
    assert after < before, f"recuperacao nao melhorou: {before:.4f} -> {after:.4f}"
    recovered = recovery_fraction(0.0, before, after)
    assert recovered > 0.1, f"recuperou apenas {recovered:.1%} da queda"


def test_recovery_stops_on_plateau(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=6), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)

    res = recover(student, logits,
                  RecoveryConfig(epochs=10, grad_accum=1, eval_every=2,
                                 plateau_patience=1, plateau_threshold=0.99,
                                 gradient_checkpointing=False),
                  device="cpu", evaluate=lambda: 42.0)
    assert res.stopped_by == "plateau"
    assert res.epochs_completed < 10


def test_recovery_respects_time_budget(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=8), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)
    res = recover(student, logits,
                  RecoveryConfig(epochs=50, grad_accum=1, gradient_checkpointing=False),
                  device="cpu")
    assert res.stopped_by in {"epochs", "time"}
    res2 = recover(student, logits,
                   RecoveryConfig(epochs=50, grad_accum=1, max_seconds=0.001,
                                  gradient_checkpointing=False),
                   device="cpu")
    assert res2.stopped_by == "time"


def test_checkpoint_written_when_metric_improves(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=4), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)
    seq = iter([10.0, 5.0, 1.0, 0.5, 0.4, 0.3])
    recover(student, logits,
            RecoveryConfig(epochs=2, grad_accum=1, eval_every=2,
                           gradient_checkpointing=False),
            device="cpu", evaluate=lambda: next(seq, 0.1),
            checkpoint_dir=tmp_path / "ck")
    assert (tmp_path / "ck" / "best.pt").is_file()
    assert (tmp_path / "ck" / "recovery.json").is_file()


# ---------------------------------------------------------------- metricas

@pytest.mark.parametrize("teacher,pruned,recovered,expected", [
    (10.0, 30.0, 14.0, 0.80),
    (10.0, 30.0, 30.0, 0.00),
    (10.0, 30.0, 10.0, 1.00),
    (10.0, 30.0, 35.0, -0.25),
    (10.0, 8.0, 8.0, 1.00),
])
def test_recovery_fraction(teacher, pruned, recovered, expected):
    assert recovery_fraction(teacher, pruned, recovered) == pytest.approx(expected)

"""Testes de funções de perda e treinamento de recuperação."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

from aguardente.arch import Arch, count_params
from aguardente.distill.loss import kd_loss
from aguardente.distill.teacher import estimate_logit_bytes, precompute_logits
from aguardente.distill.train import (
    RecoveryConfig,
    _load_checkpoint,
    _lr_at,
    _save_checkpoint_resumavel,
    recover,
)
from aguardente.errors import AguardenteError
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


# ---------------------------------------------------------------- checkpoint

def test_checkpoint_e_salvo_periodicamente(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=8), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)
    ckpt = tmp_path / "ckpt"

    vistos = []

    def registra(passo, perda):
        vistos.append((passo, (ckpt / "last.pt").is_file()))

    recover(student, logits,
            RecoveryConfig(epochs=2, grad_accum=1, checkpoint_every=2,
                           gradient_checkpointing=False),
            device="cpu", checkpoint_dir=ckpt, on_step=registra)

    marcos = [existe for passo, existe in vistos if passo > 0 and passo % 2 == 0]
    assert marcos and all(marcos)


def test_checkpoint_e_removido_ao_terminar_normalmente(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=6), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)
    ckpt = tmp_path / "ckpt"

    res = recover(student, logits,
                  RecoveryConfig(epochs=2, grad_accum=1, checkpoint_every=1,
                                 gradient_checkpointing=False),
                  device="cpu", checkpoint_dir=ckpt)

    assert res.stopped_by == "epochs"
    assert not (ckpt / "last.pt").is_file()
    assert (ckpt / "recovery.json").is_file()


def test_interrupcao_salva_checkpoint(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=6), tmp_path / "l", top_k=8)
    student = tiny_model(seed=1)
    ckpt = tmp_path / "ckpt"

    def interrompe(passo, perda):
        if passo >= 2:
            raise KeyboardInterrupt

    res = recover(student, logits,
                  RecoveryConfig(epochs=5, grad_accum=1, gradient_checkpointing=False),
                  device="cpu", checkpoint_dir=ckpt, on_step=interrompe)

    assert res.stopped_by == "interrupted"
    assert res.steps == 2
    assert (ckpt / "last.pt").is_file()


def test_retomada_processa_o_total_certo_de_passos(tmp_path):
    teacher = tiny_model()
    logits = precompute_logits(teacher, fixed_batches(n=8), tmp_path / "l", top_k=8)

    cfg = RecoveryConfig(epochs=4, grad_accum=1, gradient_checkpointing=False)
    referencia = recover(tiny_model(seed=1), logits, cfg, device="cpu")
    assert referencia.stopped_by == "epochs"

    parou_em = referencia.steps // 2

    def interrompe(passo, perda):
        if passo >= parou_em:
            raise KeyboardInterrupt

    ckpt = tmp_path / "ckpt"
    student = tiny_model(seed=1)
    parcial = recover(student, logits, cfg, device="cpu",
                      checkpoint_dir=ckpt, on_step=interrompe)
    assert parcial.stopped_by == "interrupted"
    assert parcial.steps == parou_em

    final = recover(student, logits, cfg, device="cpu", checkpoint_dir=ckpt)
    assert final.resumed_from == parou_em
    assert final.steps == referencia.steps
    assert final.stopped_by == "epochs"
    assert not (ckpt / "last.pt").is_file()


def test_retomada_preserva_estado_do_otimizador(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    student = tiny_model(seed=2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    batch = fixed_batches(n=1)[0]
    out = student(input_ids=batch["input_ids"], labels=batch["input_ids"])
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    _save_checkpoint_resumavel(student, optimizer, ckpt, step=1, lotes_feitos=1,
                               best=0.5, stale=1)

    novo_student = tiny_model(seed=2)
    novo_optimizer = torch.optim.AdamW(novo_student.parameters(), lr=1e-3)
    assert not novo_optimizer.state

    estado = _load_checkpoint(ckpt, novo_student, novo_optimizer, device="cpu")

    assert estado == {"step": 1, "lotes_feitos": 1, "best": 0.5, "stale": 1}
    assert novo_optimizer.state


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


# ---------------------------------------------------------------- novas garantias teóricas

def test_kd_loss_ignores_padding_tokens_with_mask():
    torch.manual_seed(0)
    s = torch.randn(2, 6, 64)
    v, idx = s.topk(8, dim=-1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    labels = torch.randint(0, 64, (2, 6))

    loss1 = kd_loss(s, v, idx, labels=labels, mask=mask, alpha=0.5)

    # Altera valores exclusivamente nas posições mascaradas (padding)
    s2 = s.clone()
    s2[:, 4:, :] += 100.0
    v2 = v.clone()
    v2[:, 4:, :] += 50.0
    labels2 = labels.clone()
    labels2[:, 4:] = 12

    loss2 = kd_loss(s2, v2, idx, labels=labels2, mask=mask, alpha=0.5)
    assert float(loss1) == pytest.approx(float(loss2), rel=1e-5)


def test_kd_loss_sequence_length_invariance():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 64)
    v, idx = (s + 0.5).topk(8, dim=-1)
    loss_short = kd_loss(s, v, idx, alpha=1.0)

    # Duplica ao longo da dimensão de tempo mantendo a distribuição por token
    s_long = s.repeat(1, 10, 1)
    v_long = v.repeat(1, 10, 1)
    idx_long = idx.repeat(1, 10, 1)
    loss_long = kd_loss(s_long, v_long, idx_long, alpha=1.0)

    assert float(loss_short) == pytest.approx(float(loss_long), rel=1e-4)


def test_kd_loss_fp16_numerical_stability():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 128, dtype=torch.float16)
    v = torch.randn(2, 8, 16, dtype=torch.float16)
    idx = torch.randint(0, 128, (2, 8, 16), dtype=torch.int32)
    labels = torch.randint(0, 128, (2, 8))

    loss = kd_loss(s, v, idx, labels=labels, alpha=0.9, temperature=3.0)
    assert torch.isfinite(loss).all()
    assert float(loss) > 0.0


def test_precompute_saves_and_yields_attention_mask(tmp_path):
    teacher = tiny_model()
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int8)
    batches = [{"input_ids": torch.randint(0, 256, (2, 4)), "attention_mask": mask}]
    out = precompute_logits(teacher, batches, tmp_path / "mask_shards", top_k=8)

    loaded = next(iter(out.batches()))
    assert "attention_mask" in loaded
    assert torch.equal(loaded["attention_mask"], mask)


def test_lr_at_has_min_floor():
    cfg = RecoveryConfig(learning_rate=1e-4)
    lr_end = _lr_at(100, 100, cfg)
    assert lr_end == pytest.approx(1e-5)


def test_recover_restores_best_checkpoint(tmp_path):
    teacher = tiny_model()
    batches = fixed_batches(n=4)
    logits = precompute_logits(teacher, batches, tmp_path / "l", top_k=8)

    student = tiny_model(seed=1)
    ckpt = tmp_path / "ck"
    # Simula melhora inicial (10.0) seguida de degradação posterior (50.0, 60.0, 70.0)
    seq = iter([10.0, 50.0, 60.0, 70.0])
    recover(student, logits,
            RecoveryConfig(epochs=2, grad_accum=2, eval_every=2, gradient_checkpointing=False),
            device="cpu", evaluate=lambda: next(seq), checkpoint_dir=ckpt)

    best_blob = torch.load(ckpt / "best.pt", weights_only=True)
    for k, v in student.state_dict().items():
        assert torch.equal(v, best_blob["state_dict"][k])


def test_recover_steps_trailing_batches(tmp_path):
    teacher = tiny_model()
    batches = fixed_batches(n=5)
    logits = precompute_logits(teacher, batches, tmp_path / "l", top_k=8)

    student = tiny_model(seed=1)
    res = recover(student, logits,
                  RecoveryConfig(epochs=1, grad_accum=2, gradient_checkpointing=False),
                  device="cpu")
    # 5 lotes com grad_accum=2 devem gerar 3 passos de otimizador (2 regulares + 1 residual)
    assert res.steps == 3


def test_precompute_raises_on_non_finite_logits(tmp_path):
    class NanTeacher:
        training = False

        def eval(self):
            pass

        def train(self, mode):
            pass

        def __call__(self, **kwargs):
            return type("Out", (), {"logits": torch.tensor([[[float("nan")]]])})()

    with pytest.raises(AguardenteError, match="não finitos"):
        precompute_logits(NanTeacher(), fixed_batches(n=1), tmp_path / "nan_logits", top_k=8)


def test_recover_raises_on_divergent_loss(tmp_path):
    teacher = tiny_model()
    batches = fixed_batches(n=2)
    logits = precompute_logits(teacher, batches, tmp_path / "l", top_k=8)

    class NanStudent:
        def parameters(self):
            yield torch.nn.Parameter(torch.randn(2, 2))

        def named_parameters(self):
            yield "weight", torch.nn.Parameter(torch.randn(2, 2))

        def train(self):
            pass

        def eval(self):
            pass

        def __call__(self, **kwargs):
            # Retorna logits com NaN
            return type("Out", (), {"logits": torch.full((2, 16, 256), float("nan"))})()

    with pytest.raises(AguardenteError, match="divergiu"):
        recover(NanStudent(), logits, RecoveryConfig(epochs=1), device="cpu")


def test_kd_loss_rejects_invalid_parameters():
    s = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 8)
    idx = torch.randint(0, 16, (2, 4, 8))

    with pytest.raises(ValueError, match="temperature"):
        kd_loss(s, v, idx, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        kd_loss(s, v, idx, temperature=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        kd_loss(s, v, idx, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        kd_loss(s, v, idx, alpha=1.1)


def test_kd_loss_masks_left_padding():
    s = torch.randn(1, 4, 16)
    v, idx = s.topk(8, dim=-1)
    # Left padding: token 0 é pad (0), tokens 1, 2, 3 são reais (1)
    mask = torch.tensor([[0, 1, 1, 1]])
    labels = torch.randint(0, 16, (1, 4))
    loss1 = kd_loss(s, v, idx, labels=labels, mask=mask, alpha=0.5)

    # Altera os logits e rótulo do token de preenchimento à esquerda (índice 0)
    s2 = s.clone()
    s2[:, 0, :] += 50.0
    labels2 = labels.clone()
    labels2[:, 0] = 5

    loss2 = kd_loss(s2, v, idx, labels=labels2, mask=mask, alpha=0.5)
    assert float(loss1) == pytest.approx(float(loss2), rel=1e-5)


def test_precompute_overwrites_corrupt_empty_shards(tmp_path):
    teacher = tiny_model()
    empty_shard = tmp_path / "000000.pt"
    empty_shard.touch()
    assert empty_shard.stat().st_size == 0

    batches = fixed_batches(n=1)
    out = precompute_logits(teacher, batches, tmp_path, top_k=8)

    assert empty_shard.stat().st_size > 0
    loaded = next(iter(out.batches()))
    assert "values" in loaded


def test_recover_flushes_pending_gradients_on_interrupt(tmp_path):
    teacher = tiny_model()
    batches = fixed_batches(n=3)
    logits = precompute_logits(teacher, batches, tmp_path / "l", top_k=8)

    student = tiny_model(seed=1)
    called = [0]

    def interrompe(step, loss):
        called[0] += 1
        if called[0] == 3:
            raise KeyboardInterrupt

    res = recover(
        student, logits,
        RecoveryConfig(epochs=1, grad_accum=2, gradient_checkpointing=False),
        device="cpu", on_step=interrompe, checkpoint_dir=tmp_path / "ck",
    )
    assert res.stopped_by == "interrupted"
    # Com 3 lotes e grad_accum=2, a interrupção no 3º lote deve escoar o lote residual, totalizando 2 passos
    assert res.steps == 2

"""Quality-sensitive behavior: data, optimizer semantics, reconstruction and resume."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from aguardente.calibration import make_packed_batches
from aguardente.distill.teacher import precompute_logits
from aguardente.distill.train import RecoveryConfig, recover
from aguardente.errors import AguardenteError
from aguardente.inputs import require_same_tokenizer
from tests.test_distill import tiny_model, fixed_batches


class NumberTokenizer:
    eos_token_id = pad_token_id = 0
    eos_token = pad_token = "EOS"

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(x) for x in text.split()]}


def test_packing_preserves_long_documents_and_real_eos():
    batches = list(make_packed_batches(NumberTokenizer(), ["1 2 3 4 5 6 7 8 9", "10 11"],
                                      seq_len=5, batch_size=2))
    rows = [ids[mask.bool()].tolist() for batch in batches
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]
    restored = rows[0] + [token for row in rows[1:] for token in row[1:]]
    assert restored == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 0]
    assert sum(int(b["attention_mask"].sum()) for b in batches) > len(restored)
    assert len(batches[-1]["input_ids"]) == 1  # no dropped residual batch


def test_weighted_accumulation_matches_unequal_length_full_batch(tmp_path):
    batches = fixed_batches(n=2, bs=1, seq=8)
    batches[0]["attention_mask"] = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    batches[1]["attention_mask"] = torch.ones(1, 8, dtype=torch.long)
    teacher = tiny_model()
    small = precompute_logits(teacher, batches, tmp_path / "small")
    merged = [{key: torch.cat([b[key] for b in batches]) for key in batches[0]}]
    large = precompute_logits(teacher, merged, tmp_path / "large")
    a, b = tiny_model(seed=2), tiny_model(seed=2)
    recover(a, small, RecoveryConfig(epochs=1, grad_accum=4, gradient_checkpointing=False))
    recover(b, large, RecoveryConfig(epochs=1, grad_accum=1, gradient_checkpointing=False))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-4, atol=1e-6)


def test_best_checkpoint_saves_improvement_below_patience_threshold(tmp_path):
    logits = precompute_logits(tiny_model(), fixed_batches(n=2), tmp_path / "logits")
    metrics = iter([10., 9.99, 9.98])
    student = tiny_model(seed=2)
    result = recover(student, logits, RecoveryConfig(epochs=1, grad_accum=1, eval_every=1,
                     plateau_patience=5, gradient_checkpointing=False),
                     evaluate=lambda: next(metrics), checkpoint_dir=tmp_path / "ckpt")
    assert result.best_step == 2
    assert result.best_eval == 9.98
    saved = torch.load(tmp_path / "ckpt/best.pt", weights_only=True)
    assert saved["metric"] == 9.98


def test_worsening_training_restores_initial_student_without_explicit_checkpoint(tmp_path):
    logits = precompute_logits(tiny_model(), fixed_batches(n=2), tmp_path)
    student = tiny_model(seed=2)
    before = copy.deepcopy(student.state_dict())
    metrics = iter([10., 11., 12.])
    result = recover(student, logits, RecoveryConfig(epochs=1, grad_accum=1, eval_every=1,
                     gradient_checkpointing=False), evaluate=lambda: next(metrics))
    assert result.best_step == 0
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_plateau_does_not_stop_before_consuming_the_first_epoch(tmp_path):
    batches = fixed_batches(n=6, bs=1)
    logits = precompute_logits(tiny_model(), batches, tmp_path / "logits")
    student = tiny_model(seed=2)
    before = copy.deepcopy(student.state_dict())
    result = recover(student, logits, RecoveryConfig(epochs=1, grad_accum=1,
                     eval_every=1, plateau_patience=1, gradient_checkpointing=False),
                     evaluate=lambda: 42.0)
    assert result.steps == 6
    assert result.epochs_completed == 1
    assert result.tokens_seen == 6 * 15
    assert result.stopped_by == "epochs"
    assert result.best_step == 0
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_resume_replays_uncommitted_window_with_dropout_and_shuffle(tmp_path):
    logits = precompute_logits(tiny_model(), fixed_batches(n=5, bs=1), tmp_path / "logits")
    full = tiny_model(seed=7, attention_dropout=0.2)
    resumed = copy.deepcopy(full)
    cfg = RecoveryConfig(epochs=2, grad_accum=2, seed=123, gradient_checkpointing=False)
    recover(full, logits, cfg)
    calls = 0

    def interrupt(step, loss):
        nonlocal calls
        calls += 1
        if calls == 3:  # after one committed step, inside next accumulation window
            raise KeyboardInterrupt

    partial = recover(resumed, logits, cfg, on_step=interrupt, checkpoint_dir=tmp_path / "ckpt")
    assert partial.steps == 1
    assert partial.stopped_by == "interrupted"
    torch.rand(20)  # unrelated RNG use must not affect resumed dropout
    result = recover(resumed, logits, cfg, checkpoint_dir=tmp_path / "ckpt")
    assert result.tokens_seen == 5 * 2 * 15  # fixed_batches default sequence length
    for pa, pb in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_equal_vocabulary_with_different_segmentation_is_rejected():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    base = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "!": 2}, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    teacher = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>")
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    student = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>")
    assert teacher.get_vocab() == student.get_vocab()
    with pytest.raises(AguardenteError, match="tokenizers incompatíveis"):
        require_same_tokenizer(teacher, student)


def test_data_and_recipe_content_changes_invalidate_downstream(tmp_path):
    from aguardente.pipeline import RunOptions, fingerprint
    data, recipe = tmp_path / "data.txt", tmp_path / "compression.yaml"
    data.write_text("training corpus one")
    recipe.write_text("recipe one")
    opts = RunOptions(model="org/model", out_dir=tmp_path / "run", calib_file=str(data),
                      compression_config=str(recipe))
    old = fingerprint(opts, "recover")
    data.write_text("training corpus two")
    assert fingerprint(opts, "recover") != old
    old = fingerprint(opts, "export")
    recipe.write_text("recipe two")
    assert fingerprint(opts, "export") != old


def test_nested_teacher_weight_content_changes_invalidate_logits(tmp_path):
    from aguardente.pipeline import RunOptions, fingerprint
    teacher = tmp_path / "teacher"
    (teacher / "weights").mkdir(parents=True)
    weight = teacher / "weights/shard.safetensors"
    weight.write_bytes(b"one")
    opts = RunOptions(model=str(teacher), out_dir=tmp_path / "run")
    before = fingerprint(opts, "logits")
    weight.write_bytes(b"two")
    assert fingerprint(opts, "logits") != before


def test_projection_reconstruction_reduces_residual_error():
    from types import SimpleNamespace
    from aguardente.prune.reconstruct import fit_projections
    teacher = tiny_model(hidden_size=16, intermediate_size=32, num_hidden_layers=2)
    # Correlated duplicated channels make the removed contribution reconstructible.
    with torch.no_grad():
        for layer in teacher.model.layers:
            for name in ("gate_proj", "up_proj"):
                projection = getattr(layer.mlp, name)
                projection.weight[16:].copy_(projection.weight[:16])
    src = SimpleNamespace(intermediate_size=32, num_key_value_heads=2)
    dst = SimpleNamespace(intermediate_size=16, num_key_value_heads=2)
    plan = SimpleNamespace(source=src, target=dst)
    keep = torch.arange(16).repeat(2, 1)
    batches = fixed_batches(n=4, seq=32)
    fits = fit_projections(teacher, plan, keep, [[0, 1]]*2, [0, 1], lambda: iter(batches),
                           max_batches=4, damping=0.001)
    for fit, layer in zip(fits, teacher.model.layers):
        expected = layer.mlp.down_proj.weight[:, :16] + layer.mlp.down_proj.weight[:, 16:]
        sliced_error = (layer.mlp.down_proj.weight[:, :16] - expected).norm()
        assert (fit.weight - expected).norm() < sliced_error


def test_sampled_tail_gradient_matches_full_vocabulary_kl():
    from aguardente.distill.loss import kd_loss
    teacher = torch.tensor([[[2., 1., 0., -0.5, -1.]]])
    temperature = 2.0
    values, indices = teacher.topk(2, -1)
    logz = torch.logsumexp(teacher / temperature, -1)
    probabilities = (teacher / temperature).softmax(-1)
    tail = probabilities.clone().scatter_(-1, indices, 0)
    sampled = torch.multinomial(tail.reshape(1, -1), 100_000, replacement=True,
                               generator=torch.Generator().manual_seed(17)).reshape(1, 1, -1)
    student = torch.tensor([[[1., -0.2, 0.8, 1.5, -0.7]]], requires_grad=True)
    approximate = kd_loss(student, values, indices, alpha=1, temperature=temperature,
                          teacher_logsumexp=logz, teacher_tail_indices=sampled,
                          teacher_tail_values=teacher.gather(-1, sampled))
    approximate.backward()
    gradient = student.grad.clone()
    student.grad = None
    exact = torch.nn.functional.kl_div((student / temperature).log_softmax(-1),
                                      probabilities, reduction="sum") * temperature**2
    exact.backward()
    torch.testing.assert_close(gradient, student.grad, atol=0.007, rtol=0.02)


def test_teacher_cache_does_not_reuse_other_weights_or_masks(tmp_path):
    batch = fixed_batches(n=1)[0]
    a = precompute_logits(tiny_model(seed=1), [batch], tmp_path, top_k=8, tail_samples=8)
    first = next(a.batches())
    b = precompute_logits(tiny_model(seed=2), [batch], tmp_path, top_k=8, tail_samples=8)
    second = next(b.batches())
    assert not torch.equal(first["values"], second["values"])
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch["attention_mask"][:, 8:] = 0
    c = precompute_logits(tiny_model(seed=2), [batch], tmp_path, top_k=8, tail_samples=8)
    assert torch.equal(next(c.batches())["attention_mask"], batch["attention_mask"].to(torch.int8))


def test_reused_teacher_shards_keep_coverage_and_regenerate_truncation(tmp_path):
    import json
    teacher, batches = tiny_model(), fixed_batches(n=2)
    first = precompute_logits(teacher, batches, tmp_path, top_k=8, tail_samples=8)
    before = json.loads((tmp_path / "manifest.json").read_text())
    expected = next(first.batches())["values"]
    precompute_logits(teacher, batches, tmp_path, top_k=8, tail_samples=8)
    assert json.loads((tmp_path / "manifest.json").read_text())["topk_probability_mass"] == before["topk_probability_mass"]
    (tmp_path / "000000.pt").write_bytes(b"truncated")
    with pytest.warns(UserWarning, match="regenerando"):
        repaired = precompute_logits(teacher, batches, tmp_path, top_k=8, tail_samples=8)
    torch.testing.assert_close(next(repaired.batches())["values"], expected, rtol=0, atol=0)


def test_ctrl_c_during_optimizer_update_commits_before_checkpoint(tmp_path, monkeypatch):
    import signal
    logits = precompute_logits(tiny_model(), fixed_batches(n=3, bs=1), tmp_path / "logits")
    expected = tiny_model(seed=9, attention_dropout=0.1)
    interrupted = copy.deepcopy(expected)
    cfg = RecoveryConfig(epochs=2, grad_accum=2, gradient_checkpointing=False)
    recover(expected, logits, cfg)
    original = torch.optim.AdamW.step
    calls = 0

    def step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGINT)
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    result = recover(interrupted, logits, cfg, checkpoint_dir=tmp_path / "ckpt")
    assert result.stopped_by == "interrupted"
    assert result.steps == 1
    recover(interrupted, logits, cfg, checkpoint_dir=tmp_path / "ckpt")
    for pa, pb in zip(expected.parameters(), interrupted.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_checkpoint_does_not_duplicate_tied_embeddings(monkeypatch):
    from aguardente.distill.train import _cpu_state_dict
    model = tiny_model(tie_word_embeddings=True)
    calls = []
    original = torch.Tensor.cpu

    def copy_to_cpu(tensor, *args, **kwargs):
        calls.append(tensor.data_ptr())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", copy_to_cpu)
    state = _cpu_state_dict(model)
    pointer = model.get_input_embeddings().weight.data_ptr()
    assert calls.count(pointer) == 1
    assert state["model.embed_tokens.weight"] is state["lm_head.weight"]


def test_activation_calibration_uses_only_training_documents_without_padding(tmp_path):
    import json
    from aguardente.local_export import calibration_samples
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"data": {"train": ["1 2 3"], "validation": ["99"], "test": ["98"]}}))
    samples = calibration_samples(path, NumberTokenizer(), seq_len=8)
    assert [s.tolist() for s in samples] == [[[1, 2, 3, 0]]]
    assert samples[0].dtype == torch.int32


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3"])
def test_structural_pipeline_across_decoder_families(tmp_path, family):
    from dataclasses import replace
    from transformers import AutoConfig, AutoModelForCausalLM
    from aguardente.arch import Arch, count_params
    from aguardente.plan import PrunePlan
    from aguardente.prune.scoring import score_model
    from aguardente.prune.surgery import prune_model
    cfg = AutoConfig.for_model(family, hidden_size=32, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=256, max_position_embeddings=128, tie_word_embeddings=True)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    before = sum(p.numel() for p in model.parameters())
    src = Arch.from_hf_config(cfg.to_dict())
    target = replace(src, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                     num_hidden_layers=2)
    plan = PrunePlan(src, target, count_params(target).total)
    batches = fixed_batches(n=2)
    scores = score_model(model, batches)
    prune_model(model, plan, keep_ffn=scores.top_ffn(128), keep_groups=scores.top_kv_groups(1),
                keep_layers=[0, 3])
    assert sum(p.numel() for p in model.parameters()) < before
    model.save_pretrained(tmp_path / family)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / family).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batches[0]).logits, model(**batches[0]).logits)
    logits = precompute_logits(AutoModelForCausalLM.from_config(AutoConfig.for_model(family,
        hidden_size=32, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=256, max_position_embeddings=128,
        tie_word_embeddings=True)), batches, tmp_path / "logits", tail_samples=8)
    result = recover(loaded, logits, RecoveryConfig(epochs=1, grad_accum=1, gradient_checkpointing=False))
    assert result.steps == 2
    assert all(torch.isfinite(p).all() for p in loaded.parameters())

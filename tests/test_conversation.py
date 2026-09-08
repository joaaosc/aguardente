import json
from types import SimpleNamespace

import pytest

from aguardente.conversation import assistant_supervision
from aguardente.errors import AguardenteError

torch = pytest.importorskip("torch")


class Tokenizer:
    chat_template = "fixture"
    eos_token_id = pad_token_id = ord("~")
    pad_token = eos_token = "~"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}~" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def messages(prompt="P", answer="A"):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def test_response_boundary_retains_stop_token_and_masks_prompt():
    text, mask = assistant_supervision(Tokenizer(), messages())
    assert "".join(c for c, active in zip(text, mask) if active) == "A~"
    text, mask = assistant_supervision(Tokenizer(), messages() + messages("Q", "B"))
    assert "".join(c for c, active in zip(text, mask) if active) == "A~B~"


def test_unstable_template_is_rejected_without_guessing():
    class Unstable(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return str(len(messages)) + super().apply_chat_template(messages, **kwargs)
    with pytest.raises(AguardenteError, match="prefixos estáveis"):
        assistant_supervision(Unstable(), messages())


def test_packing_long_prompt_does_not_consume_response_sample_budget():
    from aguardente.calibration import make_packed_batches
    text, mask = assistant_supervision(Tokenizer(), messages("P" * 80, "ABCDE"))
    batches = list(make_packed_batches(Tokenizer(), [text], seq_len=8, batch_size=1,
                   loss_masks={text: mask}))
    scored = []
    for batch in batches:
        ids, active = batch["input_ids"][0], batch["loss_mask"][0]
        scored.extend(chr(int(ids[i])) for i in range(1, len(ids)) if active[i])
        assert not (active.bool() & ~batch["attention_mask"][0].bool()).any()
    assert "".join(scored) == "ABCDE~"
    first = next(make_packed_batches(Tokenizer(), [text], seq_len=8, batch_size=1,
                 max_samples=1, loss_masks={text: mask}))
    assert first["loss_mask"][:, 1:].sum() > 0


def test_kd_and_supervised_gradients_exclude_prompt_predictors():
    from aguardente.distill.loss import kd_loss
    logits = torch.randn(1, 5, 8, requires_grad=True)
    teacher = torch.randn_like(logits)
    values, indices = teacher.topk(4, dim=-1)
    loss = kd_loss(logits, values, indices, torch.tensor([[1, 2, 3, 4, 5]]),
                  loss_mask=torch.tensor([[0, 0, 0, 1, 1]]), alpha=0.5)
    loss.backward()
    assert logits.grad[:, :2].count_nonzero() == 0
    assert logits.grad[:, 2:4].abs().sum() > 0
    assert logits.grad[:, -1].count_nonzero() == 0


def test_teacher_cache_regenerates_changed_supervision(tmp_path):
    from aguardente.distill.teacher import precompute_logits
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, attention_mask):
            self.calls += 1
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 8))

    model = Teacher()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4),
             "loss_mask": torch.tensor([[0, 0, 1, 1]])}
    kwargs = dict(top_k=4, cache_key="same-model", max_batches=1)
    result = precompute_logits(model, [batch], tmp_path, **kwargs)
    assert json.loads((tmp_path / "manifest.json").read_text())["valid_tokens"] == 2
    assert torch.equal(next(result.batches())["loss_mask"], batch["loss_mask"])
    precompute_logits(model, [batch], tmp_path, **kwargs)
    assert model.calls == 1
    batch["loss_mask"][0, 1] = 1
    precompute_logits(model, [batch], tmp_path, **kwargs)
    assert model.calls == 2


def test_corpus_persists_masks_and_response_perplexity(tmp_path, monkeypatch):
    from aguardente.corpus import prepare_corpus
    from aguardente.pipeline import RunOptions
    from aguardente.verify import response_perplexity
    monkeypatch.setattr("aguardente.corpus.tokenizer_identity", lambda _: "fixture")
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps({"messages": messages(str(i), "A")}) for i in range(20)))
    opts = RunOptions(model="local", out_dir=tmp_path / "run", calib_file=str(path), assistant_only=True)
    corpus = prepare_corpus(opts, Tokenizer())
    assert corpus == prepare_corpus(opts, Tokenizer())
    class Uniform(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 128))
    result = response_perplexity(Uniform(), Tokenizer(), corpus.test, corpus.loss_masks, max_length=16, device="cpu")
    assert result.value == pytest.approx(128, rel=1e-5)
    assert result.tokens == 2 * len(corpus.test)


def test_recovery_combines_response_masks_frozen_dtype_and_resume(tmp_path):
    from tests.test_distill import tiny_model
    from aguardente.distill.adapters import attach_lora
    from aguardente.distill.teacher import precompute_logits
    from aguardente.distill.train import recover, RecoveryConfig
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
             "loss_mask": torch.tensor([[0, 0, 1, 1]])}
    cache = precompute_logits(tiny_model(), [batch], tmp_path / "logits", top_k=8)
    student = tiny_model(seed=1).half()
    attach_lora(student, rank=2, alpha=4)
    frozen = {n: p.detach().clone() for n, p in student.named_parameters() if not p.requires_grad}
    cfg = RecoveryConfig(epochs=2, grad_accum=1, checkpoint_every=1,
                         learning_rate=0.01, preserve_frozen_dtype=True)
    def interrupt_after_first_update(step, loss):
        raise KeyboardInterrupt
    result = recover(student, cache, cfg, device="cpu", checkpoint_dir=tmp_path / "ckpt",
                     on_step=interrupt_after_first_update)
    assert result.stopped_by == "interrupted"
    assert result.tokens_seen == 2
    assert any(p.detach().abs().sum() > 0 for n, p in student.named_parameters() if "lora_B" in n)
    for name, parameter in student.named_parameters():
        if name in frozen:
            assert parameter.dtype == torch.float16
            assert torch.equal(parameter, frozen[name])
        else:
            assert parameter.dtype == torch.float32
    resumed = recover(student, cache, cfg, device="cpu", checkpoint_dir=tmp_path / "ckpt")
    assert resumed.resumed_from == result.steps == 1
    assert resumed.steps == 2
    assert resumed.tokens_seen == 4

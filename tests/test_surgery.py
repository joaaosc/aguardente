"""A cirurgia precisa produzir um modelo que carrega, roda e tem o shape certo.

Usa um Llama minusculo criado do zero — segundos, nao gigabytes.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

from aguardente.arch import Arch, count_params
from aguardente.plan import plan_for_target, shrink
from aguardente.prune.surgery import _default_layer_selection, prune_model
from aguardente.prune.scoring import score_model


def tiny_config(**over):
    base = dict(hidden_size=64, intermediate_size=256, num_hidden_layers=6,
                num_attention_heads=8, num_key_value_heads=4, vocab_size=512,
                max_position_embeddings=128, tie_word_embeddings=True)
    base.update(over)
    return LlamaConfig(**base)


def tiny_model(**over):
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config(**over)).eval()


def arch_of(model) -> Arch:
    return Arch.from_hf_config(model.config.to_dict())


def batches(n=3, bs=2, seq=16, vocab=512):
    for _ in range(n):
        yield {"input_ids": torch.randint(0, vocab, (bs, seq))}


def forward_ok(model, seq=16):
    out = model(input_ids=torch.randint(0, model.config.vocab_size, (2, seq)))
    assert out.logits.shape == (2, seq, model.config.vocab_size)
    assert torch.isfinite(out.logits).all(), "saida contem NaN ou Inf"
    return out.logits


def test_baseline_runs():
    forward_ok(tiny_model())


def test_prune_mlp_only_runs_and_shrinks():
    m = tiny_model()
    a = arch_of(m)
    plan = type("P", (), {})()  # plano manual: so o MLP
    from aguardente.plan import PrunePlan
    plan = PrunePlan(source=a, target=a.with_(intermediate_size=128), requested_params=0)

    before = sum(p.numel() for p in m.parameters())
    rep = prune_model(m, plan, keep_ffn=torch.arange(128))
    assert rep.params_after < before
    assert m.config.intermediate_size == 128
    forward_ok(m)


def test_prune_heads_preserves_gqa_and_runs():
    m = tiny_model()
    a = arch_of(m)
    from aguardente.plan import PrunePlan
    target = a.with_(num_key_value_heads=2, num_attention_heads=4)
    rep = prune_model(m, PrunePlan(source=a, target=target, requested_params=0),
                      keep_ffn=torch.arange(a.intermediate_size), keep_groups=[0, 2])
    assert m.config.num_key_value_heads == 2
    assert m.config.num_attention_heads == 4
    forward_ok(m)


def test_prune_layers_runs_and_reindexes():
    m = tiny_model()
    a = arch_of(m)
    from aguardente.plan import PrunePlan
    target = a.with_(num_hidden_layers=3)
    rep = prune_model(m, PrunePlan(source=a, target=target, requested_params=0),
                      keep_ffn=torch.arange(a.intermediate_size),
                      keep_layers=[0, 3, 5])
    assert m.config.num_hidden_layers == 3
    assert len(m.model.layers) == 3
    # layer_idx precisa ser reindexado, senao o cache de KV escreve na posicao errada
    assert [l.self_attn.layer_idx for l in m.model.layers] == [0, 1, 2]
    forward_ok(m)


def test_prune_all_three_axes_together():
    m = tiny_model()
    a = arch_of(m)
    plan = plan_for_target(a, int(count_params(a).total * 0.45), align=32)
    rep = prune_model(
        m, plan,
        keep_ffn=torch.arange(plan.target.intermediate_size),
        keep_groups=list(range(plan.target.num_key_value_heads)),
        keep_layers=_default_layer_selection(a.num_hidden_layers, plan.target.num_hidden_layers),
    )
    assert rep.params_after < rep.params_before
    forward_ok(m)


def test_pruned_model_round_trips_through_disk(tmp_path):
    """Sem config sincronizado, save/load falha ao casar os shapes."""
    m = tiny_model()
    a = arch_of(m)
    from aguardente.plan import PrunePlan
    target = a.with_(intermediate_size=128, num_hidden_layers=4)
    prune_model(m, PrunePlan(source=a, target=target, requested_params=0),
                keep_ffn=torch.arange(128), keep_layers=[0, 2, 4, 5])

    m.save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path).eval()
    assert reloaded.config.intermediate_size == 128
    assert reloaded.config.num_hidden_layers == 4
    forward_ok(reloaded)


def test_scoring_shapes_and_ordering():
    m = tiny_model()
    scores = score_model(m, batches(), max_batches=3)
    assert scores.ffn.shape == (256,)
    assert scores.kv_groups.shape == (4,)
    assert scores.layers.shape == (6,)
    assert torch.isfinite(scores.ffn).all()
    assert len(scores.top_ffn(64)) == 64
    assert scores.top_ffn(64).tolist() == sorted(scores.top_ffn(64).tolist())


def test_keep_layers_protects_boundaries():
    m = tiny_model()
    scores = score_model(m, batches(), max_batches=2)
    kept = scores.keep_layers(3)
    assert 0 in kept and 5 in kept, "primeira e ultima precisam sobreviver"
    assert len(kept) == 3


def test_scored_prune_end_to_end():
    """O caminho real: pontuar, planejar, cortar, rodar."""
    m = tiny_model()
    a = arch_of(m)
    scores = score_model(m, batches(), max_batches=3)
    plan = plan_for_target(a, int(count_params(a).total * 0.5), align=32)
    t = plan.target
    rep = prune_model(
        m, plan,
        keep_ffn=scores.top_ffn(t.intermediate_size),
        keep_groups=scores.top_kv_groups(t.num_key_value_heads),
        keep_layers=scores.keep_layers(t.num_hidden_layers),
    )
    assert rep.ratio > 1.0
    forward_ok(m)


def test_default_layer_selection_keeps_boundaries():
    for total in (6, 12, 36):
        for keep in (2, 3, total // 2, total - 1, total):
            sel = _default_layer_selection(total, keep)
            assert len(sel) == keep, f"{total}->{keep} deu {len(sel)}"
            assert sel == sorted(set(sel))
            if keep >= 2:
                assert sel[0] == 0 and sel[-1] == total - 1

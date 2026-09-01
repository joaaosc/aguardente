"""Testes do planejador de poda e cálculo de dimensões."""

import pytest

from aguardente.arch import Arch, count_params
from aguardente.errors import PlanImpossible
from aguardente.plan import ALIGN, plan_for_target, shrink

QWEN3_4B = Arch(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                vocab_size=151936, tie_word_embeddings=True, qk_norm=True)


def test_no_shrink_at_zero():
    assert shrink(QWEN3_4B, 0.0) == QWEN3_4B


def test_mlp_shrinks_before_heads_and_layers():
    s = shrink(QWEN3_4B, 0.20)
    assert s.intermediate_size < QWEN3_4B.intermediate_size
    assert s.num_key_value_heads == QWEN3_4B.num_key_value_heads
    assert s.num_hidden_layers == QWEN3_4B.num_hidden_layers


def test_layers_shrink_last():
    mid = shrink(QWEN3_4B, 0.70)
    assert mid.num_hidden_layers == QWEN3_4B.num_hidden_layers
    deep = shrink(QWEN3_4B, 0.95)
    assert deep.num_hidden_layers < QWEN3_4B.num_hidden_layers


def test_gqa_preserved_at_every_step():
    for i in range(101):
        s = shrink(QWEN3_4B, i / 100)
        assert s.num_attention_heads % s.num_key_value_heads == 0
        assert s.heads_per_group == QWEN3_4B.heads_per_group


def test_intermediate_stays_aligned():
    for i in range(101):
        assert shrink(QWEN3_4B, i / 100).intermediate_size % ALIGN == 0


def test_monotonic_in_t():
    counts = [count_params(shrink(QWEN3_4B, i / 50)).total for i in range(51)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


@pytest.mark.parametrize("target_b", [3.0, 2.0, 1.4, 1.0])
def test_plan_hits_target_within_tolerance(target_b):
    target = int(target_b * 1e9)
    plan = plan_for_target(QWEN3_4B, target)
    got = plan.target_params
    assert got <= target * 1.02, f"passou do alvo: {got:,} > {target:,}"
    assert got >= target * 0.90, f"cortou demais: {got:,} << {target:,}"


def test_target_above_source_is_noop():
    plan = plan_for_target(QWEN3_4B, 10_000_000_000)
    assert plan.is_noop and not plan.changes()


def test_impossible_target_raises_with_hint():
    with pytest.raises(PlanImpossible) as e:
        plan_for_target(QWEN3_4B, 100_000_000)
    assert e.value.hint


def test_changes_reports_only_moved_axes():
    plan = plan_for_target(QWEN3_4B, int(3.0e9))
    names = {c[0] for c in plan.changes()}
    assert names == {"intermediate_size"}

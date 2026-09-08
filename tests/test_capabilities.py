import json
import pytest

from aguardente.capabilities import identify, inspect_model
from aguardente.target import TargetProfile
from aguardente.errors import AguardenteError


@pytest.mark.parametrize("name,task", [
    ("LlamaForCausalLM", "causal-lm"), ("Qwen3ForCausalLM", "causal-lm"),
    ("BertForMaskedLM", "masked-lm"), ("T5ForConditionalGeneration", "encoder-decoder"),
    ("ViTForImageClassification", "classification"), ("BertModel", "features"),
])
def test_tasks_do_not_require_dense_causal_dimensions(name, task):
    assert identify({"architectures": [name]}).task == task


def test_moe_and_quantized_are_described_without_claiming_dense_surgery():
    result = identify({"architectures": ["MixtralForCausalLM"], "num_local_experts": 8,
                       "quantization_config": {"quant_method": "gptq"}})
    assert result.structure == "moe"
    assert result.strategies["structural_pruning"] == "requires-architecture-adapter"
    assert result.strategies["weight_quantization"] == "requires-quantized-loader"


def test_inspect_unknown_model_is_useful(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    result = inspect_model(str(tmp_path))
    assert result["task"] == "diffusion"
    assert "requires" in result["strategies"]["coreai_export"]


def test_target_budget_does_not_claim_actual_memory():
    result = TargetProfile().estimate(params=7_000_000_000, bits=4.5, cache_bytes=1024**3)
    assert not result["within_estimate"]
    assert not result["runtime_verified"]


@pytest.mark.parametrize("kwargs", [{"ram_gib": float("nan")}, {"reserve_gib": 8}, {"context": 0}])
def test_invalid_target(kwargs):
    with pytest.raises(AguardenteError):
        TargetProfile(**kwargs)

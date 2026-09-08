"""Identify model tasks without assuming the structure required by pruning."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from .errors import AguardenteError


@dataclass(frozen=True)
class Capabilities:
    model_type: str
    architectures: tuple[str, ...]
    task: str
    modalities: tuple[str, ...]
    structure: str
    quantized: bool
    strategies: dict[str, str]

    def to_dict(self):
        return asdict(self)


def identify(config: dict, *, task: str = "auto") -> Capabilities:
    """Reported capabilities are eligibility, never proof of successful export."""
    if not isinstance(config, dict):
        raise AguardenteError("configuração do modelo precisa ser um objeto JSON")
    arch = config.get("architectures") or []
    if not isinstance(arch, (tuple, list)) or not all(isinstance(a, str) for a in arch):
        raise AguardenteError("architectures precisa ser uma lista de nomes")
    kind = str(config.get("model_type") or config.get("_class_name") or "unknown")
    modalities = ["text"]
    if any(k in config for k in ("vision_config", "vision_tower", "visual", "image_size")):
        modalities = ["text", "image"] if any(k in config for k in ("text_config", "vocab_size")) else ["image"]
    if any(k in config for k in ("audio_config", "num_mel_bins", "sampling_rate")):
        modalities.append("audio")
    sub = next((config[k] for k in ("text_config", "language_config", "llm_config")
                if isinstance(config.get(k), dict)), config)
    structure = "dense"
    if any(sub.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        structure = "moe"
    elif any(sub.get(k) for k in ("kv_lora_rank", "qk_nope_head_dim")):
        structure = "latent-attention"
    elif any(sub.get(k) for k in ("state_size", "ssm_cfg", "mamba_d_state")) or "mamba" in kind:
        structure = "recurrent-or-hybrid"
    names = " ".join(arch)
    inferred = "unknown"
    if "ForCausalLM" in names:
        inferred = "causal-lm"
    elif "ForMaskedLM" in names:
        inferred = "masked-lm"
    elif "ForSequenceClassification" in names or "ForImageClassification" in names:
        inferred = "classification"
    elif config.get("is_encoder_decoder") or "ConditionalGeneration" in names:
        inferred = "encoder-decoder"
    elif "Diffusion" in kind or "_diffusers_version" in config:
        inferred = "diffusion"
    elif arch and all(a.endswith("Model") for a in arch):
        inferred = "features"
    actual = inferred if task == "auto" else task
    from .arch import _quantized
    quantized = bool(_quantized(sub) or _quantized(config))
    if quantized:
        quant = "requires-quantized-loader"
    else:
        quant = "requires-task-adapter-and-export-validation"
    # Surgery additionally requires actual projection shapes and an architecture
    # adapter. Do not assert support from a model_type string alone.
    prune = "requires-projection-inspection" if actual == "causal-lm" and structure == "dense" and not quantized else "requires-architecture-adapter"
    return Capabilities(kind, tuple(arch), actual, tuple(dict.fromkeys(modalities)),
                        structure, quantized, {
                            "weight_quantization": quant,
                            "palettization": quant,
                            "lora": "requires-linear-modules" if not quantized else "requires-quantized-loader",
                            "structural_pruning": prune,
                            "logit_distillation": "requires-matching-output-semantics",
                            "coreai_export": "requires-graph-and-native-validation",
                        })


def inspect_model(ref: str, *, task: str = "auto") -> dict:
    path = Path(ref).expanduser()
    revision = None
    if path.exists():
        file = path if path.is_file() else path / "config.json"
        if not file.exists():
            file = path / "model_index.json"
        if not file.is_file():
            raise AguardenteError("modelo local sem config.json ou model_index.json; forneça um adaptador de tarefa")
        cfg = json.loads(file.read_text())
    else:
        from .fetch import validate_model_id
        from .probe import _get_json
        validate_model_id(ref)
        metadata = _get_json(f"https://huggingface.co/api/models/{ref}")
        revision = metadata.get("sha")
        if not revision:
            raise AguardenteError("o Hub não informou uma revisão imutável do modelo")
        files = {s["rfilename"] for s in metadata.get("siblings", [])}
        name = "config.json" if "config.json" in files else "model_index.json"
        cfg = _get_json(f"https://huggingface.co/{ref}/resolve/{revision}/{name}")
    return {"model": str(path.resolve()) if path.exists() else ref,
            "revision": revision, **identify(cfg, task=task).to_dict()}

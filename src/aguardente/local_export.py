"""Export a local checkpoint through Apple's model and conversion APIs.

No synthetic Hub repository, monkeypatch, network lookup or non-strict weight
loading. The upstream from_hf loader accepts local Transformers directories.
"""
from pathlib import Path
import logging
import tempfile
import json
import copy


def calibration_samples(path, tokenizer, *, limit=16, seq_len=256):
    """Only training documents calibrate activation ranges; exclude padding."""
    from .calibration import make_packed_batches
    if limit < 1:
        raise ValueError("calibration_samples must be positive")
    data = json.loads(Path(path).read_text())
    texts = data["data"]["train"]
    if not texts or not all(isinstance(text, str) for text in texts):
        raise ValueError("Calibration corpus has no training text")
    samples = []
    for batch in make_packed_batches(tokenizer, texts, batch_size=1,
                                      seq_len=seq_len, max_samples=limit):
        samples.append(batch["input_ids"][:, batch["attention_mask"][0].bool()].int())
    return samples


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from coreai_models.llm.export import build_parser, _resolve_export_config
    from coreai_models.models.registry import get_model_entry
    from coreai_models.export.presets import get_preset
    from coreai_models.export.compression import quantize_for_export, split_compression_config
    from coreai_models.export.macos import export_macos_model
    from coreai_models.export.bundle import bundle_llm_asset
    from coreai_models.export.metadata import build_aimodel_metadata

    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    parser.add_argument("--calibration-corpus", help="snapshot de corpus do aguardente; usa somente treino")
    parser.add_argument("--calibration-samples", type=int, default=16)
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve(strict=True))
    # The Apple preset supplies operator exclusions and layout; changing the
    # weight bitwidth is a coreai-opt configuration, not a different converter.
    int8 = args.compression == "8bit"
    if int8:
        args.compression = "4bit"
    config = _resolve_export_config(args)
    if int8:
        config.compression = "8bit"
    if config.variant != "macOS":
        raise ValueError("Local export currently supports macOS; iOS needs separate validation")
    hf_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.max_context_length:
        if config.max_context_length > hf_config.max_position_embeddings:
            raise ValueError("Requested context exceeds the trained context")
        hf_config.max_position_embeddings = config.max_context_length
    dtype = getattr(torch, config.compute_precision)
    from .textonly import EQUIVALENT
    entry = get_model_entry(EQUIVALENT.get(hf_config.model_type, hf_config.model_type))
    if entry.macos_class is None:
        raise ValueError(f"No macOS converter for {hf_config.model_type}")
    if args.dry_run:
        print(config)
        print(f"Local architecture accepted: {hf_config.model_type}")
        return
    if config.compression_config_object is not None:
        quant, palette = split_compression_config(config.compression_config_object)
        preset = {"torch_quantization_config": quant, "torch_palettization_config": palette}
    else:
        preset = copy.deepcopy(get_preset("4bit" if int8 else config.compression))
        if int8:
            def promote_weights(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "dtype" and child == "int4":
                            value[key] = "int8"
                        else:
                            promote_weights(child)
                elif isinstance(value, list):
                    for child in value:
                        promote_weights(child)
            promote_weights(preset)
    if preset.get("torch_palettization_config"):
        raise ValueError("macOS local export requires a quantization preset")
    name = args.output_name or f"{Path(args.model).name}_{config.compression}_dynamic"
    bundle = Path(args.output_dir) / name
    asset = bundle / f"{name}.aimodel"
    if asset.exists():
        raise FileExistsError(f"Output already exists: {asset}")

    with tempfile.TemporaryDirectory(prefix="aguardente-export-") as tmp:
        local_ref = args.model
        if hf_config.model_type == "llama":
            from .probe import probe_local
            from .textonly import build_config
            p = probe_local(args.model)
            emitted = build_config(json.loads((Path(args.model) / "config.json").read_text()),
                                   p.arch, p.layout)
            canonical = Path(tmp) / "canonical"
            canonical.mkdir()
            for f in Path(args.model).iterdir():
                if f.is_file() and f.name != "config.json":
                    (canonical / f.name).symlink_to(f.resolve())
            (canonical / "config.json").write_text(json.dumps(emitted.config))
            local_ref = str(canonical)
            hf_config = AutoConfig.from_pretrained(local_ref, local_files_only=True)
            if config.max_context_length:
                hf_config.max_position_embeddings = config.max_context_length
        model = entry.macos_class.from_hf(
            local_ref, target_dtype=dtype,
            max_context_length=config.max_context_length, mmap_path=tmp + "/weights",
        ).eval()
        quant = preset.get("torch_quantization_config")
        if quant is not None:
            from coreai_models.export.compression import get_c4
            def calibration_data():
                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
                if args.calibration_corpus:
                    return calibration_samples(args.calibration_corpus, tokenizer,
                        limit=args.calibration_samples, seq_len=min(256, hf_config.max_position_embeddings))
                return get_c4(tokenizer)
            model = quantize_for_export(
                model, hf_config, dtype, dict(quant),
                calibration_data_fn=calibration_data,
                mmap_dir=tmp + "/quantized",
            )
        program = export_macos_model(model, hf_config, config)
        bundle.mkdir(parents=True, exist_ok=True)
        program.save_asset(asset, build_aimodel_metadata(args.model))
        bundle_llm_asset(bundle, args.model, hf_config, config.compression, name)
        print(f"Export complete: {bundle}", flush=True)


if __name__ == "__main__":
    main()

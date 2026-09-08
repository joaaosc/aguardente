"""Numerical and KV-cache checks of a macOS LLM asset on the real Core AI runtime.

These engineering tolerances detect conversion faults; they do not certify
language quality. Compare quantized outputs with the recovered FP32 checkpoint.
"""
from pathlib import Path
import argparse
import asyncio
import json
import math
import time

PROMPTS = [
    "The Earth moves around the Sun. The Moon moves around the Earth. Together, these objects form part of",
    "A good way to learn a new language is to practice every day. You can read books, listen to music, and",
    "Era uma vez uma pequena cidade perto do mar. Todas as manhãs, os moradores saíam de casa para",
]


async def validate(model_dir, asset, *, compression="4bit", precision="float16", max_new_tokens=32, test_text=None, max_ppl_ratio=1.05):
    import numpy as np
    import torch
    from .swift_runtime import SwiftRuntime
    from .loading import load_causal_lm

    start = time.perf_counter()
    reference, tokenizer = load_causal_lm(str(model_dir), device="cpu", dtype=torch.float32)
    cfg = reference.config
    meta = json.loads((Path(asset).parent / "metadata.json").read_text())
    context = int(meta["language"]["max_context_length"])
    shape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, context,
             getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    runtime = SwiftRuntime(asset, shape, precision)
    try:
        eos = reference.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos if eos is not None else tokenizer.eos_token_id])
        fn = "main"
        print(f"Runtime loaded: {runtime.function_names}", flush=True)

        def new_state():
            return runtime.new_state()

        async def call(function, ids, total, state):
            return await runtime.call(function, ids, total, state)

        comparisons = []
        generations = []
        failures = []
        for prompt in PROMPTS:
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            if len(ids) + max_new_tokens >= context:
                raise ValueError("Context is too short for the validation prompts")
            state = new_state()
            prediction = await call(fn, ids, len(ids), state)
            actual = prediction["logits"].numpy().astype(np.float32)
            with torch.no_grad():
                expected = reference(input_ids=torch.tensor([ids])).logits.numpy()
            if not np.isfinite(actual).all():
                raise ValueError("Core AI produced non-finite logits")
            diff = actual.astype(np.float64) - expected
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
            cosine = float(np.sum(actual.astype(np.float64) * expected) /
                           (np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(expected.astype(np.float64))))
            entry = {"prompt": prompt, "tokens": len(ids), "rmse": rmse,
                     "relative_rmse": rmse / max(rms, 1e-12), "cosine": cosine,
                     "top1_agreement": float(np.mean(actual.argmax(-1) == expected.argmax(-1))),
                     "psnr_db": 20 * math.log10(float(np.max(np.abs(expected))) / max(rmse, 1e-12))}

            # Verify the prefill entrypoint and an incremental decode at the SAME
            # prefix, independently of autoregressive sampling divergence.
            split_state = new_state()
            prefill = "prefill" if "prefill" in runtime.function_names else fn
            await call(prefill, ids[:-1], len(ids) - 1, split_state)
            incremental = (await call(fn, ids[-1:], len(ids), split_state))["logits"].numpy()
            cache_error = float(np.max(np.abs(incremental.astype(np.float32) - actual[:, -1:])))
            entry["prefill_decode_max_abs_error"] = cache_error
            await runtime.release_state(split_state)
            if not np.allclose(incremental, actual[:, -1:], atol=0.15, rtol=0.02):
                raise ValueError(f"KV cache/prefill differs from full forward: {cache_error}")
            tolerance = 0.15 if compression != "none" else 0.02
            if entry["relative_rmse"] > tolerance or cosine < (0.98 if compression != "none" else 0.999):
                failures.append(f"Numerical tolerance exceeded for prompt {len(comparisons) + 1}")
            comparisons.append(entry)
            print(json.dumps(entry), flush=True)

            generated = []
            for step in range(max_new_tokens):
                token = int(actual[0, -1].argmax())
                generated.append(token)
                if token in eos_ids:
                    break
                actual = (await call(fn, [token], len(ids) + step + 1, state))["logits"].numpy()
                if not np.isfinite(actual).all():
                    raise ValueError("Non-finite logits during autoregressive decoding")
            await runtime.release_state(state)
            with torch.no_grad():
                baseline = reference.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                              max_new_tokens=max_new_tokens,
                                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generations.append({"prompt": prompt, "coreai": tokenizer.decode(generated),
                                "pytorch": tokenizer.decode(baseline[0, len(ids):])})

        from .calibration import load_texts
        text = (test_text if test_text is not None else "\n\n".join(load_texts(split="test", limit=512, min_chars=1)))[:6000]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        ref_nll = core_nll = 0.0
        counted = 0
        prev_end = 0
        window_size = min(256, context)
        stride = max(1, window_size // 2)
        for begin in range(0, len(ids), stride):
            end = min(begin + window_size, len(ids))
            if end <= prev_end:
                break
            window = ids[begin:end]
            state = new_state()
            actual = (await call(fn, window, len(window), state))["logits"].numpy().astype(np.float32)
            await runtime.release_state(state)
            if not np.isfinite(actual).all():
                raise ValueError("Non-finite logits during perplexity evaluation")
            with torch.no_grad():
                expected = reference(input_ids=torch.tensor([window])).logits
            first_target = max(1, prev_end - begin)
            labels = torch.tensor(window[first_target:])
            for values, is_core in ((torch.from_numpy(actual), True), (expected, False)):
                nll = float(torch.nn.functional.cross_entropy(
                    values[0, first_target-1:-1].float(), labels, reduction="sum"))
                if is_core:
                    core_nll += nll
                else:
                    ref_nll += nll
            counted += len(labels)
            prev_end = end
            if end == len(ids):
                break
        if not counted:
            raise ValueError("No evaluation tokens")
        perplexity = {"pytorch": math.exp(ref_nll / counted), "coreai": math.exp(core_nll / counted),
                      "tokens": counted, "window": window_size, "stride": stride,
                      "dataset": "provided held-out text, first 6000 chars" if test_text is not None else "Salesforce/wikitext wikitext-2-raw-v1 test, first 6000 chars"}
        perplexity["ratio"] = perplexity["coreai"] / perplexity["pytorch"]
        perplexity["max_ratio"] = max_ppl_ratio
        if perplexity["ratio"] > max_ppl_ratio:
            failures.append(f"Perplexity ratio {perplexity['ratio']:.4f} exceeds {max_ppl_ratio}")
        print(json.dumps({"perplexity": perplexity}), flush=True)
        # Exercise multiple prefill calls followed by decode near the declared
        # context limit (up to 1024 tokens to bound validation memory).
        import inspect
        length = min(context - 1, len(ids), 1024)
        prefix = ids[:length]
        state = new_state()
        prefill = "prefill" if "prefill" in runtime.function_names else fn
        for offset in range(0, length - 1, 128):
            chunk = prefix[offset:min(offset + 128, length - 1)]
            await call(prefill, chunk, offset + len(chunk), state)
        last = (await call(fn, prefix[-1:], length, state))["logits"].numpy().astype(np.float32)
        await runtime.release_state(state)
        kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(reference.forward).parameters else {}
        with torch.no_grad():
            source_last = reference(input_ids=torch.tensor([prefix]), **kwargs).logits[:, -1:].numpy()
        error = float(np.sqrt(np.mean((last - source_last)**2))) / max(float(np.sqrt(np.mean(source_last**2))), 1e-12)
        long_context = {"tokens": length, "prefill_chunk": 128, "relative_rmse": error,
                        "declared_context": context}
        if not np.isfinite(last).all() or error > (0.02 if compression == "none" else 0.15):
            failures.append("Chunked prefill/decode exceeds numerical tolerance")
        return {"asset": str(asset), "source": str(model_dir), "compression": compression,
                "passed": not failures, "failures": failures,
                "preferred_compute": "gpu",
                "comparisons": comparisons, "generations": generations, "perplexity": perplexity,
                "chunked_prefill": long_context,
                "seconds": time.perf_counter() - start,
                "scope": "Three prompts, held-out perplexity, chunked prefill/decode up to 1024 tokens; not a capability benchmark"}
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("asset")
    parser.add_argument("--compression", default="4bit")
    parser.add_argument("--precision", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--report", required=True)
    parser.add_argument("--test-text-file")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.05)
    args = parser.parse_args()
    try:
        text = Path(args.test_text_file).read_text() if args.test_text_file else None
        result = asyncio.run(validate(args.model, args.asset, compression=args.compression,
                                      precision=args.precision, test_text=text, max_ppl_ratio=args.max_ppl_ratio))
    except Exception as exc:
        result = {"passed": False, "asset": args.asset, "failure_kind": "runtime_error",
                  "failures": [f"{type(exc).__name__}: {exc}"]}
    Path(args.report).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit("Runtime validation failed; diagnostics were saved to " + args.report)


if __name__ == "__main__":
    main()

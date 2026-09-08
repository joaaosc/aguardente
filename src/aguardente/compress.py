"""Recipe search for explicit model tasks, with isolated native validation."""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import uuid
import shutil
import math
import statistics

from .errors import AguardenteError
from .evaluation import QualityPolicy, output_fidelity
from .inputs import local_identity, file_identity
from .state import run_lock
from .tasks import ModelTask, build_hf_task, load_factory
from .target import TargetProfile

RECIPES = ("none", "w8", "w4-block32", "w4-block128", "mixed4-8", "palette4", "palette6")


def _artifact_identity(path):
    from .inputs import file_identity
    path = Path(path)
    if path.is_file():
        return {path.name: file_identity(path)}
    if not path.is_dir():
        raise AguardenteError("artefato ausente")
    return {str(p.relative_to(path)): file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def _task(spec):
    import torch
    torch.manual_seed(spec["seed"])
    if spec.get("adapter"):
        task = load_factory(spec["adapter"])(spec["model"], spec["data"])
    else:
        task = build_hf_task(spec["model"], spec["data"], task=spec["task"], seq_len=spec["seq_len"])
    if not isinstance(task, ModelTask):
        raise AguardenteError("factory deve devolver ModelTask")
    task.validate()
    if not callable(task.score):
        raise AguardenteError("adaptador precisa fornecer uma métrica da tarefa; paridade sozinha não aprova qualidade")
    task.model.eval()
    return task


def _predict(model, batches, expected_outputs=None):
    import torch
    results = []
    with torch.no_grad():
        for batch in batches:
            outputs = model(*batch)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            if not isinstance(outputs, (tuple, list)) or not all(isinstance(x, torch.Tensor) for x in outputs):
                raise AguardenteError("forward do adaptador deve devolver um tensor ou tupla de tensores")
            if expected_outputs is not None and len(outputs) != expected_outputs:
                raise AguardenteError("quantidade de saídas difere dos nomes declarados")
            if any(x.dtype not in (torch.float32, torch.float16) for x in outputs):
                raise AguardenteError("runtime estático exige saídas float32 ou float16; adapte explicitamente")
            results.append(tuple(x.detach().cpu().float().numpy() for x in outputs))
    return results


def _save_outputs(path, outputs):
    import numpy as np
    np.savez(path, **{f"sample_{i}_output_{j}": x for i, row in enumerate(outputs) for j, x in enumerate(row)})


def _load_outputs(path, task, split):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return [tuple(data[f"sample_{i}_output_{j}"] for j in range(len(task.output_names)))
                for i in range(len(task.samples[split]))]


def _prepare(task, recipe):
    """Explicit weight-only recipes; no automatic fallback after SDK errors."""
    import torch
    if recipe == "none":
        return task.model, None, []
    overrides = []
    if recipe.startswith("palette"):
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
        config = getattr(KMeansPalettizerConfig.presets, "w" + recipe[-1])()
        for name, module in task.model.named_modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim >= 2 and weight.shape[0] % 16:
                config.set_module_name(name, None)
                overrides.append({"module": name, "precision": str(weight.dtype),
                                  "reason": "indivisible-palette-group"})
        compressor_class = KMeansPalettizer
    else:
        from coreai_opt.quantization import Quantizer, QuantizerConfig, ModuleQuantizerConfig, ExecutionMode
        if recipe == "w8":
            config = QuantizerConfig.presets.w8(execution_mode=ExecutionMode.EAGER)
        else:
            block = 128 if recipe == "w4-block128" else 32
            config = QuantizerConfig.presets.w4_per_block(block_size=block, execution_mode=ExecutionMode.EAGER)
            for name, module in task.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    reason = None
                    if module.weight.shape[1] % block:
                        reason = "indivisible-input-dimension"
                    elif recipe == "mixed4-8" and name.rsplit(".", 1)[-1] in {"lm_head", "classifier", "score", "q_proj", "k_proj"}:
                        reason = "explicit-sensitive-module-recipe"
                    if reason:
                        config.set_module_name(name, ModuleQuantizerConfig.presets.w8())
                        overrides.append({"module": name, "precision": "int8", "reason": reason})
        compressor_class = Quantizer
    # Normalization scales and scalar/vector parameters are kept in their
    # original precision. A matrix quantization axis has no meaning for them.
    for name, module in task.model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim < 2:
            config.set_module_name(name, None)
            overrides.append({"module": name, "precision": str(weight.dtype),
                              "reason": "non-matrix-weight"})
    compressor = compressor_class(task.model, config)
    prepared = compressor.prepare(task.samples["calibration"][0])
    # These presets compress weights only. Calibration inputs exercise the graph;
    # this does not claim activation-aware weight reconstruction or AWQ.
    _predict(prepared, task.samples["calibration"], len(task.output_names))
    return prepared, compressor, overrides


def _export(task, model, compressor, directory):
    import torch
    from coreai_torch import TorchConverter
    from coreai_torch._decomp import get_decomp_table
    if compressor:
        model = compressor.finalize()
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, task.samples["calibration"][0])
        exported = exported.run_decompositions(get_decomp_table())
        if exported.graph_signature.buffers_to_mutate or exported.graph_signature.user_inputs_to_mutate:
            raise AguardenteError("adaptador estático não aceita mutações ocultas; exponha estado como entradas e saídas")
        program = (TorchConverter().add_exported_program(exported,
                    input_names=task.input_names, output_names=task.output_names).to_coreai())
        program.optimize()
    directory = Path(tempfile.mkdtemp(prefix="pending-", dir=directory))
    asset = directory / "model.aimodel"
    program.save_asset(asset)
    processor = dict(task.preprocessing or {})
    tokenizer = processor.pop("tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    _write(directory / "interface.json", {"input_names": task.input_names,
        "output_names": task.output_names, "shapes": [list(t.shape) for t in task.samples["calibration"][0]],
        "dtypes": [str(t.dtype) for t in task.samples["calibration"][0]],
        "task": task.description, "preprocessing": processor,
        "stateful": False, "minimum_macos": "27.0"})
    return asset


def _asset_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size


def _native(task, asset, split):
    from .task_runtime import TaskRuntime
    runtime = TaskRuntime(asset)
    try:
        outputs = [runtime.predict(batch, task.input_names, task.output_names) for batch in task.samples[split]]
        return outputs, {"load_seconds": runtime.load_seconds, "call_seconds": runtime.timings,
                         "host_peak_rss_bytes": runtime.host_peak_rss_bytes,
                         "memory_scope": "host process RSS; not total unified GPU memory"}
    finally:
        runtime.close()


def _resource_failures(spec, timing):
    target = TargetProfile(**spec["target"])
    return ["target_host_rss"] if timing["host_peak_rss_bytes"] > target.available_bytes else []


def pareto_candidates(candidates):
    def measures(row):
        runtime = row["runtime"]
        metrics = row.get("native_metrics", {})
        return (row["bytes"], runtime["host_peak_rss_bytes"], statistics.median(runtime["call_seconds"]),
                row["native_fidelity"]["relative_rmse"],
                *(value if key.startswith("loss:") else -value for key, value in sorted(metrics.items())))
    vectors = [measures(row) for row in candidates]
    return [row for i, row in enumerate(candidates) if not any(
        all(a <= b for a, b in zip(other, vectors[i], strict=True)) and
        any(a < b for a, b in zip(other, vectors[i], strict=True))
        for j, other in enumerate(vectors) if j != i)]


def worker(spec_path, recipe, *, final=False):
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec["out"])
    directory = root / recipe
    directory.mkdir(exist_ok=True)
    task = _task(spec)
    policy = QualityPolicy(**spec["policy"])
    started = time.monotonic()
    if recipe == "baseline":
        result = {"status": "reference", "metrics": {}, "references": {}}
        for split in ("validation", "test"):
            outputs = _predict(task.model, task.samples[split], len(task.output_names))
            output_fidelity(outputs, outputs)  # reject nonfinite references
            score = task.score(outputs, split)
            if not score or policy.failures({"relative_rmse": 0}, score, score):
                raise AguardenteError("métricas da referência vazias ou inválidas")
            result["metrics"][split] = score
            path = directory / f"{split}.npz"
            _save_outputs(path, outputs)
            result["references"][split] = local_identity(str(path))
        result["parameters"] = sum(p.numel() for p in task.model.parameters())
        result["samples"] = {s: len(b) for s, b in task.samples.items()}
        _write(directory / "result.json", result)
        return
    baseline = json.loads((root / "baseline/result.json").read_text())
    split = "test" if final else "validation"
    reference_path = root / "baseline" / f"{split}.npz"
    if local_identity(str(reference_path)) != baseline["references"][split]:
        raise AguardenteError("cache da referência foi alterado")
    reference = _load_outputs(reference_path, task, split)
    if final:
        asset = directory / "bundle/model.aimodel"
        record = json.loads((directory / "result.json").read_text())
        if record["artifact_identity"] != _artifact_identity(asset.parent):
            raise AguardenteError("artefato do candidato foi alterado")
        native, timing = _native(task, asset, split)
        fidelity = output_fidelity(reference, native)
        score = task.score(native, split)
        failures = policy.failures(fidelity, baseline["metrics"][split], score)
        failures.extend(_resource_failures(spec, timing))
        _write(directory / "test.json", {"status": "rejected" if failures else "accepted", "failures": failures,
                 "fidelity": fidelity, "metrics": score, "runtime": timing})
        return
    model, compressor, overrides = _prepare(task, recipe)
    outputs = _predict(model, task.samples[split], len(task.output_names))
    fidelity = output_fidelity(reference, outputs)
    score = task.score(outputs, split)
    failures = policy.failures(fidelity, baseline["metrics"][split], score)
    record = {"recipe": recipe, "status": "rejected", "torch_fidelity": fidelity,
              "torch_metrics": score, "failures": failures, "overrides": overrides}
    if not failures:
        asset = _export(task, model, compressor, directory)
        size = _asset_bytes(asset)
        record.update(bytes=size)
        if spec.get("max_asset_bytes") and size > spec["max_asset_bytes"]:
            failures.append("max_asset_bytes")
        else:
            native, timing = _native(task, asset, split)
            fidelity = output_fidelity(reference, native)
            score = task.score(native, split)
            failures.extend(policy.failures(fidelity, baseline["metrics"][split], score))
            failures.extend(_resource_failures(spec, timing))
            record.update(native_fidelity=fidelity, native_metrics=score, runtime=timing)
        if not failures:
            record["status"] = "validated"
        bundle = directory / "bundle"
        if bundle.exists():
            bundle.rename(directory / f"previous-{uuid.uuid4().hex}")
        asset.parent.rename(bundle)
        record["artifact_identity"] = _artifact_identity(bundle)
    record["seconds"] = time.monotonic() - started
    _write(directory / "result.json", record)


def search(spec: dict, *, report=print):
    root = Path(spec["out"]).expanduser().resolve()
    model_path = Path(spec["model"]).expanduser().resolve()
    if model_path.is_dir() and (root == model_path or model_path in root.parents):
        raise AguardenteError("a saída deve ficar fora do diretório do checkpoint original")
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(spec, out=str(root))
    spec.setdefault("target", TargetProfile(context=spec["seq_len"]).to_dict())
    TargetProfile(**spec["target"])
    spec.setdefault("select", "size")
    if spec["select"] not in {"size", "latency", "fidelity"}:
        raise AguardenteError("objetivo de seleção inválido")
    if spec["task"] == "auto":
        from .capabilities import inspect_model
        spec["task"] = "custom" if spec.get("adapter") else inspect_model(spec["model"])["task"]
    if not spec.get("adapter") and spec["task"] not in {"causal-lm", "masked-lm", "classification"}:
        raise AguardenteError("a tarefa exige --adapter ou --task explícito compatível com os pesos")
    if not Path(spec["model"]).exists():
        raise AguardenteError("compress exige checkpoint local para fixar pesos; use aguardente fetch primeiro")
    if not spec["recipes"] or any(r not in RECIPES for r in spec["recipes"]):
        raise AguardenteError("receitas de compressão inválidas")
    spec["recipes"] = list(dict.fromkeys(spec["recipes"]))
    if not Path(spec["data"]).is_file():
        raise AguardenteError("arquivo de dados não encontrado")
    if spec.get("timeout", 1800) < 1:
        raise AguardenteError("timeout precisa ser positivo")
    spec["identity"] = {"version": 2, "model": _artifact_identity(spec["model"]), "data": local_identity(spec["data"]),
        "adapter": local_identity(spec["adapter"].rpartition(":")[0]) if spec.get("adapter") else None}
    spec["identity"]["implementation"] = {name: file_identity(Path(__file__).with_name(name))
        for name in ("compress.py", "tasks.py", "conversation.py", "evaluation.py", "task_runtime.py", "task_runtime.swift")}
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ("torch", "transformers", "coreai-opt", "coreai-torch"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    spec["identity"]["packages"] = versions
    if not spec.get("adapter"):
        from .probe import read_safetensors_header, _summarize_tensors
        from .budget import Machine, Budget
        parameters = sum(_summarize_tensors(read_safetensors_header(str(p)))[0]
                         for p in model_path.glob("*.safetensors"))
        if not parameters:
            raise AguardenteError("checkpoint sem pesos safetensors inspecionáveis")
        budget = Budget.for_machine(Machine.detect(root))
        # Current generic preparation is FP32 on CPU and may duplicate weights
        # during finalization/export. This is separate from compressed inference.
        if parameters * 8 > budget.ram_bytes:
            raise AguardenteError("a preparação genérica FP32 excede a RAM estimada; use uma máquina maior ou adaptador com carregamento por blocos")
        if parameters * 4 * (len(spec["recipes"]) + 1) > shutil.disk_usage(root).free:
            raise AguardenteError("disco insuficiente para os candidatos; reduza a quantidade de receitas ou mude o destino")
    spec_path = root / "spec.json"
    with run_lock(root):
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise AguardenteError("configuração/dados/pesos mudaram; use outro diretório para preservar a execução")
        _write(spec_path, spec)
        _write(root / "selection.json", {"status": "running"})
        try:
            return _search_candidates(root, spec_path, spec, report)
        except BaseException as exc:
            previous = json.loads((root / "selection.json").read_text())
            if previous.get("status") != "rejected":
                _write(root / "selection.json", {"status": "failed", "error": str(exc) or type(exc).__name__})
            raise


def _search_candidates(root, spec_path, spec, report):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    def execute(recipe, final=False):
        report(f"compressão: {recipe}{' · teste final' if final else ''}")
        args = [sys.executable, "-m", "aguardente.compress", str(spec_path), recipe]
        if final:
            args.append("test")
        with (root / f"{recipe}{'-test' if final else ''}.log").open("w") as log:
            process = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=spec.get("timeout", 1800))
        if process.returncode:
            raise AguardenteError(f"{recipe} falhou (código {process.returncode}); consulte o log em {root}")
        if not final:
            _write(root / recipe / "receipt.json", file_identity(root / recipe / "result.json"))

    def completed(recipe):
        result = root / recipe / "result.json"
        receipt = root / recipe / "receipt.json"
        if not result.exists() or not receipt.exists():
            return False
        if file_identity(result) != json.loads(receipt.read_text()):
            raise AguardenteError(f"relatório de {recipe} foi alterado; use outro diretório")
        return json.loads(result.read_text()).get("status") != "failed"

    if not completed("baseline"):
        execute("baseline")
    candidates = []
    for recipe in spec["recipes"]:
        result = root / recipe / "result.json"
        if not completed(recipe):
            try:
                execute(recipe)
            except (AguardenteError, subprocess.TimeoutExpired) as error:
                # A broken recipe is a failed candidate, never a validated
                # asset. Independent recipes may still meet the original gates.
                result.parent.mkdir(exist_ok=True)
                _write(result, {"recipe": recipe, "status": "failed", "failures": [str(error)]})
                _write(result.parent / "receipt.json", file_identity(result))
        row = json.loads(result.read_text())
        candidates.append(row)
        report(f"{recipe}: {row['status']} {row.get('failures', [])}")
    valid = [r for r in candidates if r["status"] == "validated"]
    if not valid:
        _write(root / "selection.json", {"status": "rejected", "candidates": candidates})
        raise AguardenteError("nenhuma receita cumpriu as tolerâncias e o orçamento declarados")
    frontier = pareto_candidates(valid)
    objective = {"size": lambda r: r["bytes"],
                 "latency": lambda r: statistics.median(r["runtime"]["call_seconds"]),
                 "fidelity": lambda r: r["native_fidelity"]["relative_rmse"]}[spec["select"]]
    chosen = min(frontier, key=lambda r: (objective(r), r["bytes"], r["native_fidelity"]["relative_rmse"]))
    execute(chosen["recipe"], final=True)
    test = json.loads((root / chosen["recipe"] / "test.json").read_text())
    selection = {"status": test["status"], "selected": chosen["recipe"],
                 "asset": f"{chosen['recipe']}/bundle/model.aimodel", "candidates": candidates, "test": test,
                 "quality_scope": "provided task metrics and numerical fidelity on supplied samples",
                 "pareto_recipes": [row["recipe"] for row in frontier], "objective": spec["select"],
                 "target": spec["target"],
                 "runtime_memory_verified": False}
    _write(root / "selection.json", selection)
    if test["status"] != "accepted":
        raise AguardenteError("candidato selecionado reprovou no teste final; não selecionar outro usando o mesmo teste")
    return selection

def cmd_compress(args):
    if args.max_asset_mib is not None and (not math.isfinite(args.max_asset_mib) or args.max_asset_mib <= 0):
        raise AguardenteError("max-asset-mib precisa ser positivo e finito")
    if args.quality_policy:
        policy = QualityPolicy(**json.loads(Path(args.quality_policy).read_text()))
    else:
        policy = QualityPolicy(args.max_relative_rmse, args.max_loss_ratio, args.max_score_drop)
    spec = {"model": str(Path(args.model).expanduser().resolve()), "data": str(Path(args.data).expanduser().resolve()),
            "out": args.out, "task": args.task, "adapter": args.adapter, "seq_len": args.seq_len,
            "recipes": args.recipes.split(","), "seed": args.seed, "policy": asdict(policy),
            "max_asset_bytes": int(args.max_asset_mib * 1024**2) if args.max_asset_mib is not None else None,
            "timeout": args.timeout}
    spec["select"] = args.select
    spec["target"] = TargetProfile(ram_gib=args.target_ram_gib, reserve_gib=args.target_reserve_gib,
                                    context=args.seq_len, max_asset_bytes=spec["max_asset_bytes"]).to_dict()
    if spec["max_asset_bytes"] is not None and spec["max_asset_bytes"] <= 0:
        raise AguardenteError("max-asset-mib precisa ser positivo")
    result = search(spec)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2], final=len(sys.argv) > 3)

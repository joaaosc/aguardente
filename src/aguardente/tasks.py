"""Explicit task adapters keep compression independent of model architecture.

A custom factory is trusted local code selected by the CLI user. It receives
model_ref and data_path and returns ModelTask with a fresh model on every call.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .errors import AguardenteError

SPLITS = ("calibration", "validation", "test")


@dataclass
class ModelTask:
    model: Any
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    samples: dict[str, list[tuple]]
    score: Callable[[list[tuple], str], dict[str, float]] | None = None
    description: str = "custom"
    preprocessing: dict | None = None

    def validate(self):
        import torch
        if not isinstance(self.model, torch.nn.Module):
            raise AguardenteError("adaptador precisa fornecer um torch.nn.Module")
        for names in (self.input_names, self.output_names):
            if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n.isidentifier() for n in names):
                raise AguardenteError("nomes de entradas/saídas inválidos ou duplicados")
        seen = {}
        contract = None
        for split in SPLITS:
            if not self.samples.get(split):
                raise AguardenteError(f"adaptador sem amostras reais de {split}")
            for batch in self.samples[split]:
                if len(batch) != len(self.input_names) or not all(isinstance(x, torch.Tensor) for x in batch):
                    raise AguardenteError("entradas precisam corresponder aos nomes declarados")
                shape = tuple((tuple(x.shape), str(x.dtype)) for x in batch)
                if contract is not None and shape != contract:
                    raise AguardenteError("o adaptador estático exige formas/dtypes iguais; agrupe ou prepare as entradas")
                contract = shape
                digest = hashlib.sha256(str(shape).encode())
                for tensor in batch:
                    if tensor.dtype not in (torch.float32, torch.float16, torch.int32):
                        raise AguardenteError("runtime estático aceita entradas float32, float16 ou int32; adapte o dtype explicitamente")
                    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                        raise AguardenteError("entrada vazia ou não finita")
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                key = digest.hexdigest()
                if key in seen and seen[key] != split:
                    raise AguardenteError(f"contaminação entre {seen[key]} e {split}: entradas idênticas")
                seen[key] = split
        return self


def load_factory(spec: str):
    path, separator, name = spec.rpartition(":")
    if not separator or not name.isidentifier():
        raise AguardenteError("adaptador deve ser arquivo.py:factory")
    file = Path(path).expanduser().resolve(strict=True)
    if file.suffix != ".py":
        raise AguardenteError("adaptador precisa ser um arquivo Python local")
    module_spec = importlib.util.spec_from_file_location("aguardente_user_task", file)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    factory = getattr(module, name, None)
    if not callable(factory):
        raise AguardenteError(f"factory {name!r} ausente no adaptador")
    return factory


def build_hf_task(model_ref: str, data_path: str, *, task: str, seq_len: int = 64) -> ModelTask:
    """Stateless HF adapter: causal/MLM/classification outputs, no head synthesis."""
    import torch
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForMaskedLM, AutoModelForSequenceClassification)
    classes = {"causal-lm": AutoModelForCausalLM, "masked-lm": AutoModelForMaskedLM,
               "classification": AutoModelForSequenceClassification}
    if task not in classes:
        raise AguardenteError("esta tarefa requer --adapter arquivo.py:factory")
    if seq_len < 2:
        raise AguardenteError("seq_len precisa ser >= 2")
    local = Path(model_ref).is_dir()
    config = AutoConfig.from_pretrained(model_ref, local_files_only=local, trust_remote_code=False)
    if config.to_dict().get("quantization_config"):
        raise AguardenteError("checkpoint pré-quantizado exige um adaptador compatível com seu formato")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise AguardenteError("tokenizer sem pad/EOS; adaptador deve definir padding explicitamente")
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
    samples = {s: [] for s in SPLITS}
    labels = {s: [] for s in SPLITS}
    truncated = 0
    input_names = ("input_ids", "attention_mask")
    for row in rows:
        response_mask = None
        split = row.get("split")
        if split not in samples:
            raise AguardenteError("cada exemplo deve declarar split calibration, validation ou test")
        if "messages" in row:
            if not getattr(tokenizer, "chat_template", None):
                raise AguardenteError("mensagens exigem um template de conversa do tokenizer")
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                    add_generation_prompt=False, **row.get("template_kwargs", {}))
            special = False
            if task == "causal-lm":
                from .conversation import assistant_supervision
                text, response_mask = assistant_supervision(tokenizer, row["messages"], row.get("template_kwargs"))
        else:
            text = row.get("text")
            special = True
        if not isinstance(text, str) or not text.strip():
            raise AguardenteError("cada exemplo exige text ou messages não vazios")
        length = len(tokenizer(text, add_special_tokens=special).input_ids)
        truncated += int(length > seq_len)
        encoded = tokenizer(text, add_special_tokens=special, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=seq_len)
        batch = tuple(encoded[n].to(torch.int32) for n in input_names)
        samples[split].append(batch)
        if task == "causal-lm":
            target = encoded["input_ids"][:, 1:].clone()
            valid = encoded["attention_mask"][:, 1:].bool() & encoded["attention_mask"][:, :-1].bool()
            if response_mask is not None:
                # Honor both padding and truncation sides without treating EOS
                # token IDs as padding. The attention mask defines real tokens.
                active_length = int(encoded["attention_mask"].sum())
                response_mask = response_mask[-seq_len:] if tokenizer.truncation_side == "left" else response_mask[:seq_len]
                supervised = torch.zeros_like(encoded["input_ids"])
                positions = encoded["attention_mask"].bool()
                supervised[positions] = torch.tensor(response_mask[:active_length])
                valid &= supervised[:, 1:].bool()
            target[~valid] = -100
        elif task == "classification":
            if "label" not in row or not isinstance(row["label"], int):
                raise AguardenteError("classificação exige label inteiro em cada exemplo")
            target = torch.tensor([row["label"]])
        else:
            targets = row.get("label_ids")
            if targets is None:
                raise AguardenteError("masked-lm exige label_ids com -100 nas posições não avaliadas")
            target = torch.tensor([targets], dtype=torch.long)
            if target.shape != encoded["input_ids"].shape:
                raise AguardenteError("label_ids deve ter seq_len posições")
        if not (target != -100).any():
            raise AguardenteError("exemplo sem alvos válidos para avaliação")
        labels[split].append(target)

    model, info = classes[task].from_pretrained(model_ref, local_files_only=local,
                      trust_remote_code=False, dtype=torch.float32,
                      attn_implementation="eager", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise AguardenteError(f"pesos treinados ausentes/incompatíveis para {task}: {info}")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    class Outputs(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.model = module

        def forward(self, input_ids, attention_mask):
            return (self.model(input_ids=input_ids, attention_mask=attention_mask).logits,)

    def score(outputs, split):
        loss = 0.0
        tokens = correct = 0
        for output, target in zip(outputs, labels[split], strict=True):
            logits = torch.as_tensor(output[0]).float()
            if task == "causal-lm":
                logits = logits[:, :-1]
            flat = logits.reshape(-1, logits.shape[-1])
            label = target.reshape(-1)
            loss += torch.nn.functional.cross_entropy(flat, label, reduction="sum", ignore_index=-100).item()
            valid = label != -100
            tokens += int(valid.sum())
            correct += int((flat.argmax(-1)[valid] == label[valid]).sum())
        metrics = {"loss:cross_entropy": loss / tokens}
        if task != "causal-lm":
            metrics["score:accuracy"] = correct / tokens
        return metrics

    return ModelTask(Outputs(model), input_names, ("logits",), samples, score,
            description=task, preprocessing={"tokenizer": tokenizer, "seq_len": seq_len,
                 "truncated_samples": truncated, "unexpected_keys": sorted(info.get("unexpected_keys", []))}).validate()

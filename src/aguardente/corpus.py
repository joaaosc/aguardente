"""Persist the actual training and held-out documents before producing logits."""
import hashlib
import json
import random
from dataclasses import dataclass

from .calibration import DEFAULT_CONFIG, load_texts, load_texts_from_file
from .errors import AguardenteError
from .inputs import local_identity, tokenizer_identity


def _unique(texts):
    return list(dict.fromkeys(t.strip() for t in texts if t.strip()))


@dataclass
class Corpus:
    train: list[str]
    validation: list[str]
    test: list[str]
    fingerprint: str


def prepare_corpus(opts, tokenizer) -> Corpus:
    fields = ("calib_dataset", "calib_config", "calib_split", "text_column", "calib_file", "eval_file", "seed", "seq_len", "calib_batches", "logit_batches")
    spec = {k: getattr(opts, k) for k in fields}
    spec.update(version=1, files={k: local_identity(getattr(opts, k)) for k in ("calib_file", "eval_file")},
                tokenizer=tokenizer_identity(tokenizer))
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = opts.out_dir / "data" / "corpus.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["spec_hash"] == key:
            data = saved["data"]
            digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            if digest != saved["content_hash"]:
                raise AguardenteError("corpus persistido foi modificado; use um novo diretório de execução")
            return Corpus(**data, fingerprint=digest)
    limit = max(opts.calib_batches, opts.logit_batches) * 16 + 64
    if opts.calib_file:
        train = load_texts_from_file(opts.calib_file, limit=limit, min_chars=1,
                                    seed=opts.seed, tokenizer=tokenizer)
    else:
        train = load_texts(dataset=opts.calib_dataset,
                           config=opts.calib_config if opts.calib_dataset else DEFAULT_CONFIG,
                           split=opts.calib_split, limit=limit, min_chars=1,
                           seed=opts.seed, text_column=opts.text_column)
    train = _unique(train)
    random.Random(opts.seed).shuffle(train)
    if opts.calib_file or opts.calib_dataset:
        if len(train) < 20:
            raise AguardenteError("calibração exige pelo menos 20 documentos distintos para separar treino, validação e teste",
                                  hint="Forneça mais documentos em TXT/JSONL ou um dataset adequado; não reutilize treino como avaliação.")
        held = max(2, len(train) // 10)
        validation, test, train = train[:held], train[held:2*held], train[2*held:]
    else:
        validation = load_texts(split="validation", limit=512, min_chars=1)
        test = load_texts(split="test", limit=512, min_chars=1)
    if opts.eval_file:
        validation = load_texts_from_file(opts.eval_file, limit=512, min_chars=1, tokenizer=tokenizer)
    validation = _unique(validation)
    val_set = set(validation)
    test = [t for t in _unique(test) if t not in val_set]
    held_set = val_set | set(test)
    train = [t for t in train if t not in held_set]
    if not train or not test or not validation:
        raise AguardenteError("treino, validação e teste precisam de documentos distintos não vazios")
    data = dict(train=train, validation=validation, test=test)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(spec=spec, spec_hash=key, content_hash=digest, data=data), ensure_ascii=False))
    temporary.replace(path)
    return Corpus(**data, fingerprint=digest)

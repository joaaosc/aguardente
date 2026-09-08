"""Held-out data must stay disjoint and reproducible when a run is resumed."""
import json

import pytest

from aguardente import corpus
from aguardente.calibration import load_texts_from_file
from aguardente.errors import AguardenteError
from aguardente.pipeline import RunOptions


def test_custom_corpus_is_disjoint_deduplicated_and_snapshot_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "tokenizer_identity", lambda _: "tokenizer")
    documents = [f"Document {i} with enough distinct content." for i in range(40)]
    path = tmp_path / "train.txt"
    path.write_text("\n\n".join(documents + documents))
    opts = RunOptions(model="org/model", out_dir=tmp_path / "run", calib_file=str(path))
    prepared = corpus.prepare_corpus(opts, None)
    assert len(prepared.train) == 32
    assert len(prepared.validation) == len(prepared.test) == 4
    assert not set(prepared.train) & (set(prepared.validation) | set(prepared.test))
    assert not set(prepared.validation) & set(prepared.test)
    monkeypatch.setattr(corpus, "load_texts_from_file", lambda *a, **kw: pytest.fail("must reuse snapshot"))
    assert corpus.prepare_corpus(opts, None) == prepared


def test_tampered_corpus_snapshot_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "tokenizer_identity", lambda _: "tokenizer")
    path = tmp_path / "train.txt"
    path.write_text("\n\n".join(f"Document {i}" for i in range(30)))
    opts = RunOptions(model="org/model", out_dir=tmp_path / "run", calib_file=str(path))
    corpus.prepare_corpus(opts, None)
    snapshot = opts.out_dir / "data/corpus.json"
    data = json.loads(snapshot.read_text())
    data["data"]["train"].append("contamination")
    snapshot.write_text(json.dumps(data))
    with pytest.raises(AguardenteError, match="modificado"):
        corpus.prepare_corpus(opts, None)


def test_jsonl_renders_messages_with_teacher_template(tmp_path):
    class Tokenizer:
        chat_template = "trained template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert not tokenize and not add_generation_prompt
            return "<user>" + messages[0]["content"] + "</user>"

    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "Hello"}]}))
    assert load_texts_from_file(str(path), tokenizer=Tokenizer(), min_chars=1) == ["<user>Hello</user>"]
    with pytest.raises(AguardenteError, match="chat_template"):
        load_texts_from_file(str(path), min_chars=1)


def test_short_jsonl_never_becomes_raw_json_training_text(tmp_path):
    path = tmp_path / "short.jsonl"
    path.write_text(json.dumps({"text": "short", "metadata": "x" * 500}))
    with pytest.raises(AguardenteError, match="não contém"):
        load_texts_from_file(str(path), min_chars=100)

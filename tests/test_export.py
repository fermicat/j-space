# Phenomenon-level assertions on the exported demo data: the static page's
# claims must hold in the data it ships. Runs against viz/data/*.json (fast,
# no model load); regenerate with `uv run python -m jspace_repro.run_readout`.
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "viz" / "data"

REQUIRED_KEYS = {
    "prompt", "tokens", "layers", "top_n", "top_ids", "top_ranks", "tracked",
    "ranks", "pinned", "vocab", "vocab_size", "answer", "jspace_layers",
    "jspace", "title_zh", "note_zh", "focus",
}


def load(slug):
    return json.loads((DATA / f"{slug}.json").read_text(encoding="utf-8"))


def all_slugs():
    return [e["slug"] for e in json.loads((DATA / "index.json").read_text())]


@pytest.mark.parametrize("slug", all_slugs())
def test_schema_and_shapes(slug):
    d = load(slug)
    assert REQUIRED_KEYS <= set(d)
    seq, n_layers = len(d["tokens"]), len(d["layers"])
    assert len(d["top_ids"]) == seq
    assert len(d["top_ids"][0]) == n_layers
    assert len(d["top_ids"][0][0]) == d["top_n"]
    for tid, rows in d["ranks"].items():
        assert len(rows) == seq and len(rows[0]) == n_layers
    assert d["pinned"], f"{slug}: no pinned concept survived tokenization"


@pytest.mark.parametrize("slug", all_slugs())
def test_jspace_codes_valid(slug):
    d = load(slug)
    assert d["jspace"], "empty j-space grid"
    for key, cell in d["jspace"].items():
        assert len(cell["atoms"]) <= 25, f"{slug} {key}: sparsity budget exceeded"
        assert all(c > 0 for _, c in cell["atoms"]), f"{slug} {key}: negative coeff"
        assert 0.0 <= cell["ev"] <= 1.0


def best_rank(d, token_str, pos=-1):
    ids = [int(k) for k, v in d["vocab"].items() if v == token_str and k in d["ranks"]]
    assert ids, f"{token_str!r} not tracked"
    return min(min(d["ranks"][str(i)][pos]) for i in ids)


def test_silent_italy_in_boot_demo():
    # The page claims ' Italy' enters the top-10 despite never being output.
    d = load("multihop-boot")
    assert best_rank(d, " Italy") < 10
    assert "Italy" not in d["answer"]


def test_thought_vs_speech_dissociation():
    # ' Italy' near the top internally; the spoken answer is 莫斯科.
    d = load("capital-cn")
    assert best_rank(d, " Italy") <= 2
    assert "莫斯科" in d["answer"]
    assert "罗马" not in d["answer"]


def test_ocean_modulation_reaches_top():
    d = load("mod-topic")
    ocean = [" ocean", " Ocean", " fish", " whale", " sea", " shark",
             " dolphin", " marine"]
    assert min(best_rank(d, w) for w in ocean) == 0

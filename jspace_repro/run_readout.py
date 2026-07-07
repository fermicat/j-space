# Phase B export: compute the readout payload for every demo and write
# viz/data/<slug>.json plus an index. Run: uv run python -m jspace_repro.run_readout
import json
import time
from pathlib import Path

from jlens.examples import resolve_prompt

from jspace_repro.core import compute_readout, load
from jspace_repro.demos import DEMOS

OUT_DIR = Path(__file__).resolve().parents[1] / "viz" / "data"


def main() -> None:
    model, lens, tok, gloss = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for demo in DEMOS:
        t0 = time.time()
        prompt = resolve_prompt(demo.example, tok)
        payload = compute_readout(
            model, lens, tok, gloss, prompt,
            concepts=demo.concepts,
            meta={
                "slug": demo.slug,
                "title_en": demo.title_en,
                "title_zh": demo.title_zh,
                "desc_en": demo.desc_en,
                "desc_zh": demo.desc_zh,
                "note_en": demo.note_en,
                "note_zh": demo.note_zh,
                "focus": {"pos": demo.focus_pos, "layer": demo.focus_layer},
            },
        )
        out = OUT_DIR / f"{demo.slug}.json"
        out.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        index.append({"slug": demo.slug, "title_en": demo.title_en,
                      "title_zh": demo.title_zh})
        print(
            f"{demo.slug:16} {out.stat().st_size/1e3:8.0f} KB  "
            f"seq={len(payload['tokens'])}  {time.time()-t0:5.1f}s"
        )
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

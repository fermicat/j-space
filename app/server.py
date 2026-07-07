# Phase D: local interactive server. Loads model + lens once; serves the same
# visualization page in live mode plus POST /api/readout for arbitrary prompts.
# Run: uv run uvicorn app.server:app --port 8123
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from jspace_repro.core import MAX_PROMPT_TOKENS, compute_readout, load

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "viz"

app = FastAPI(title="j-space live readout")
_model = _lens = _tok = _gloss = None
_lock = threading.Lock()  # one readout at a time; MPS is not reentrant here


class ReadoutRequest(BaseModel):
    prompt: str
    chat: bool = True  # wrap in the chat template (matches the demo prompts)


@app.on_event("startup")
def _load() -> None:
    global _model, _lens, _tok, _gloss
    _model, _lens, _tok, _gloss = load()


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    template = (VIZ / "template.html").read_text(encoding="utf-8")
    app_js = (VIZ / "app.js").read_text(encoding="utf-8")
    index = json.loads((VIZ / "data" / "index.json").read_text(encoding="utf-8"))
    demos = [
        json.loads((VIZ / "data" / f"{e['slug']}.json").read_text(encoding="utf-8"))
        for e in index
    ]
    boot = json.dumps(
        {"mode": "live", "demos": demos}, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return (
        '<!doctype html>\n<html lang="zh-CN">\n'
        + template.replace("__BOOT__", boot).replace("__APP_JS__", app_js)
        + "\n</html>"
    )


@app.post("/api/readout")
def readout(req: ReadoutRequest) -> dict:
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(422, "prompt 不能为空 / prompt must be non-empty")
    n_tokens = len(_tok.encode(prompt))
    if n_tokens > MAX_PROMPT_TOKENS:
        raise HTTPException(
            422,
            f"prompt 过长:{n_tokens} tokens,上限 {MAX_PROMPT_TOKENS} / prompt too long",
        )
    if req.chat:
        prompt = _tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    with _lock:
        return compute_readout(
            _model, _lens, _tok, _gloss, prompt,
            meta={"slug": "custom", "focus": {"pos": -1, "layer": 12}},
        )

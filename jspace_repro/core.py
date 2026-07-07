# Shared computation for the j-space visualization: load model + lens once,
# turn a prompt into the full readout payload the pages render (top-k grid,
# tracked-token ranks, j-space sparse codes, greedy answer).
import gzip
import json
from pathlib import Path

import torch
import transformers

import jlens
from jlens.vis import _meaningful_token_mask, compute_slice

from jspace_repro.jspace import decompose_position

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
GLOSS_PATH = Path(__file__).resolve().parents[1] / (
    "vendor/jacobian-lens/assets/qwen_gloss.json.gz"
)

TOP_N = 8  # top tokens kept per (position, layer) cell
JSPACE_LAYER_STRIDE = 4  # decompose every Nth lens layer
JSPACE_K = 25  # paper's workspace sparsity budget
JSPACE_CANDIDATES = 2000
MAX_TRACKED = 100
MAX_PROMPT_TOKENS = 512


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load() -> tuple:
    """(model, lens, tokenizer, gloss) — call once per process."""
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device())
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    gloss = {int(k): v for k, v in json.load(gzip.open(GLOSS_PATH)).items()}
    return model, lens, tok, gloss


def _single_token_ids(tok, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return ids


@torch.no_grad()
def _greedy_answer(model, tok, prompt: str, max_new_tokens: int = 16) -> str:
    ids = model.encode(prompt, max_length=MAX_PROMPT_TOKENS)
    out = model._hf_model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def _jspace_grid(model, lens, prompt: str, layers: list[int]) -> dict:
    """j-space sparse code at every (position, layer in `layers`).

    Keys are "pos,layer"; values {atoms: [[token_id, coeff], ...], ev: float}.
    """
    from jlens.hooks import ActivationRecorder

    input_ids = model.encode(prompt, max_length=MAX_PROMPT_TOKENS)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {l: recorder.activations[l].detach() for l in layers}

    mask = _meaningful_token_mask(
        model.tokenizer, model._lm_head.weight.shape[0], torch.device("cpu")
    )
    grid = {}
    for layer in layers:
        residuals = activations[layer][0].float()  # [seq, d_model]
        transported = lens.transport(residuals, layer)
        logits = model.unembed(transported).float().cpu()  # [seq, vocab]
        for pos in range(residuals.shape[0]):
            code = decompose_position(
                model, lens, residuals[pos], logits[pos], layer,
                n_candidates=JSPACE_CANDIDATES, k=JSPACE_K, display_mask=mask,
            )
            grid[f"{pos},{layer}"] = {
                "atoms": [
                    [t, round(c, 3)]
                    for t, c in zip(code.token_ids, code.coefficients)
                    if c > 0
                ],
                "ev": round(code.explained_variance, 4),
            }
    return grid


@torch.no_grad()
def compute_readout(
    model,
    lens,
    tok,
    gloss: dict[int, str],
    prompt: str,
    *,
    concepts: list[str] | None = None,
    with_jspace: bool = True,
    meta: dict | None = None,
) -> dict:
    """Full payload for one prompt, in the schema both pages render."""
    pinned = set(_single_token_ids(tok, concepts or []))
    slice_data = compute_slice(
        model,
        lens,
        prompt,
        top_n=TOP_N,
        max_tracked=MAX_TRACKED,
        pinned_token_ids=pinned,
        mask_display=True,
        max_seq_len=MAX_PROMPT_TOKENS,
    )

    jspace_layers = lens.source_layers[::JSPACE_LAYER_STRIDE]
    if lens.source_layers[-1] not in jspace_layers:
        jspace_layers = jspace_layers + [lens.source_layers[-1]]

    payload = {
        "prompt": prompt,
        "tokens": slice_data.context_token_strs,
        "layers": slice_data.layers,
        "top_n": TOP_N,
        "top_ids": slice_data.top_ids.tolist(),  # [pos][layer][k]
        "top_ranks": slice_data.top_ranks.tolist(),
        "tracked": slice_data.tracked_token_ids,
        "ranks": {
            str(tid): slice_data.rank_tensor[:, :, i].tolist()
            for i, tid in enumerate(slice_data.tracked_token_ids)
        },
        "pinned": sorted(pinned & set(slice_data.tracked_token_ids)),
        "vocab": {str(k): v for k, v in slice_data.vocab_fragment.items()},
        "vocab_size": slice_data.vocab_size,
        "answer": _greedy_answer(model, tok, prompt),
        "jspace_layers": jspace_layers if with_jspace else [],
        "jspace": _jspace_grid(model, lens, prompt, jspace_layers)
        if with_jspace
        else {},
    }
    vocab_ids = {int(k) for k in payload["vocab"]}
    for cell in payload["jspace"].values():
        vocab_ids.update(t for t, _ in cell["atoms"])
    payload["vocab"] = {
        str(t): tok.decode([t], clean_up_tokenization_spaces=False)
        for t in vocab_ids
    }
    payload["gloss"] = {str(t): gloss[t] for t in vocab_ids if t in gloss}
    if meta:
        payload.update(meta)
    return payload

# Screen the two voluntary-modulation demos (paper protocol): while the
# assistant writes a fixed prefilled sentence, does the side-task concept
# (ocean creatures / the arithmetic result 7) surface in the lens readout
# at the sentence positions?
import torch
import transformers

import jlens
from jlens.examples import EXAMPLES, resolve_prompt

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
hf = hf.to(DEVICE)
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

by_slug = {e.slug: e for e in EXAMPLES}


def single_ids(strs):
    out = []
    for s in strs:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            out.append(enc[0])
    return out


CASES = [
    ("mod-topic", by_slug["modulation-topic"],
     [" ocean", " fish", " whale", " sea", " Ocean", " shark", " dolphin"]),
    ("mod-arith", by_slug["modulation-arithmetic"],
     ["7", " 7", " seven", "seven"]),
]

for name, ex, concepts in CASES:
    prompt = resolve_prompt(ex, tok)
    ids = single_ids(concepts)
    n_prefill = len(tok.encode(ex.assistant_prefill, add_special_tokens=False))
    lens_logits, _, input_ids = lens.apply(model, prompt)
    seq_len = input_ids.shape[1]
    span = range(seq_len - n_prefill, seq_len)  # the prefilled sentence
    best = (10**9, -1, -1)
    for layer, logits in lens_logits.items():
        for pos in span:
            ranks = torch.argsort(logits[pos], descending=True)
            inv = torch.empty_like(ranks)
            inv[ranks] = torch.arange(len(ranks))
            r = min(int(inv[i]) for i in ids)
            if r < best[0]:
                best = (r, layer, pos)
    r, layer, pos = best
    tok_str = tok.decode([input_ids[0, pos]])
    print(f"{name:10} best concept rank={r} @L{layer} pos={pos} (token {tok_str!r})")

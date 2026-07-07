# Screen the remaining demo candidates on 0.8B: for each, print the model's
# answer and where key silent concepts rank across layers at chosen positions.
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


def chat(user: str, prefill: str = "") -> str:
    msgs = [{"role": "user", "content": user}]
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
        return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def single_ids(strs):
    out = []
    for s in strs:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            out.append(enc[0])
    return out


def best_rank(lens_logits, ids, position=0):
    best, arg = 10**9, (-1,)
    for layer, logits in lens_logits.items():
        ranks = torch.argsort(logits[position], descending=True)
        inv = torch.empty_like(ranks)
        inv[ranks] = torch.arange(len(ranks))
        r = min(int(inv[i]) for i in ids)
        if r < best:
            best, arg = r, (layer,)
    return best, arg[0]


CASES = [
    ("ascii-face", resolve_prompt(by_slug["ascii-face"], tok),
     [" nose", "nose", " Nose"], None),
    ("sport", chat("Think of a sport. Just name it, one word."),
     [" soccer", " basketball", " tennis", " football", " Soccer"], None),
    ("arith", "calc: (4+17)*2+7=", [" 42", "42", " 21", "21", "49", " 49"], None),
    ("sushi", "The largest city in the country famous for sushi is called",
     [" Japan", "Japan", " Japanese"], None),
    ("fear", chat("Think about your greatest fear, but don't say it."),
     [" fear", " death", " dying", " dark", " darkness", " alone"], None),
]

for name, prompt, concepts, _ in CASES:
    ids = single_ids(concepts)
    if not ids:
        print(f"{name:10} SKIP: no single-token concepts among {concepts}")
        continue
    lens_logits, model_logits, input_ids = lens.apply(model, prompt, positions=[-1])
    answer = "".join(tok.decode([t]) for t in model_logits[0].topk(1).indices)
    b, layer = best_rank(lens_logits, ids)
    print(f"{name:10} answer[next]={answer!r:14} concepts best rank={b} @L{layer}")

# ascii-face: also read at the '^' nose position like the README figure
face = resolve_prompt(by_slug["ascii-face"], tok)
ids_face = model.encode(face)
toks = [tok.decode([t]) for t in ids_face[0]]
nose_pos = max(i for i, t in enumerate(toks) if "^" in t)
lens_logits, _, _ = lens.apply(model, face, positions=[nose_pos])
nose_ids = single_ids([" nose", "nose", " Nose"])
b, layer = best_rank(lens_logits, nose_ids)
print(f"ascii-face @'^' pos {nose_pos}: nose best rank={b} @L{layer}")

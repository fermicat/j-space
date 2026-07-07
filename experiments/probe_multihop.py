# Probe: where (if anywhere) does the 0.8B model hold the silent "Italy"
# intermediate on the multihop prompt? Read out the last few positions with
# word-like display masking, and check the model's actual answer.
import torch
import transformers

import jlens
from jlens.examples import EXAMPLES
from jlens.vis import _meaningful_token_mask

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
hf = hf.to(DEVICE)
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

prompt = next(e for e in EXAMPLES if e.slug == "multihop").prompt
print(repr(prompt))

ids = model.encode(prompt)
toks = [tok.decode([t]) for t in ids[0]]
print("last tokens:", toks[-6:])

lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[-1])
mask = _meaningful_token_mask(tok, lens_logits[0].shape[-1], torch.device("cpu"))

italy_ids = [tok.encode(s, add_special_tokens=False) for s in [" Italy", "Italy", " Italian"]]
italy_flat = [i[0] for i in italy_ids if len(i) == 1]

for layer in sorted(lens_logits):
    logits = lens_logits[layer][0]
    masked = logits.masked_fill(~mask, float("-inf"))
    top = [tok.decode([t]) for t in masked.topk(8).indices]
    ranks = torch.argsort(logits, descending=True)
    inv = torch.empty_like(ranks)
    inv[ranks] = torch.arange(len(ranks))
    italy_rank = min(int(inv[i]) for i in italy_flat)
    print(f"L{layer:>2} italy_rank={italy_rank:>6}  {top}")

print("model answer:", [tok.decode([t]) for t in model_logits[0].topk(8).indices])

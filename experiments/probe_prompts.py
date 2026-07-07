# Which prompts does 0.8B actually do multi-hop on? For each candidate,
# check the model's answer and the best cross-layer lens rank of the silent
# intermediate at the last positions.
import torch
import transformers

import jlens

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
hf = hf.to(DEVICE)
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)


def chat(user: str, prefill: str = "") -> str:
    msgs = [{"role": "user", "content": user}]
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
        return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


CASES = [
    # (name, prompt, intermediate concept tokens, expected answer fragment)
    ("boot-raw", "Fact: The currency used in the country shaped like a boot is",
     [" Italy", "Italy", " Italian"], "lira/euro"),
    ("boot-chat", chat("In the country shaped like a boot, what currency do they use? Answer with only the currency name."),
     [" Italy", "Italy", " Italian"], "euro"),
    ("planet-raw", "Fact: The color of the fourth planet from the Sun is",
     [" Mars", "Mars"], "red"),
    ("planet-chat", chat("What color is the fourth planet from the Sun? Answer with only the color."),
     [" Mars", "Mars"], "red"),
    ("capital-cn", chat("形状像靴子的国家的首都是哪座城市?只回答城市名。"),
     ["意大利", " Italy", "Italy"], "罗马/Rome"),
]

for name, prompt, inter_strs, expect in CASES:
    inter_ids = []
    for s in inter_strs:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            inter_ids.append(enc[0])
    lens_logits, model_logits, input_ids = lens.apply(model, prompt, positions=[-1])
    answer = [tok.decode([t]) for t in model_logits[0].topk(5).indices]

    best_rank, best_layer = 10**9, -1
    for layer, logits in lens_logits.items():
        ranks = torch.argsort(logits[0], descending=True)
        inv = torch.empty_like(ranks)
        inv[ranks] = torch.arange(len(ranks))
        r = min(int(inv[i]) for i in inter_ids)
        if r < best_rank:
            best_rank, best_layer = r, layer
    print(f"{name:12} expect={expect:10} answer={answer}")
    print(f"{'':12} intermediate best rank={best_rank} @L{best_layer}")

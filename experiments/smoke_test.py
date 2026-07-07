# Phase A smoke test: load Qwen3.5-0.8B + pre-fitted lens, reproduce the
# README two-hop example ("country shaped like a boot" -> silent "Italy").
import torch
import transformers

import jlens

MODEL_ID = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(DEVICE)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    print(f"device={DEVICE}  {lens}")

    prompt = "Fact: The currency used in the country shaped like a boot is"
    lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[-2])
    for layer, logits in sorted(lens_logits.items()):
        top = [tok.decode([t]) for t in logits[0].topk(5).indices]
        print(f"L{layer:>2}  {top}")
    print("model:", [tok.decode([t]) for t in model_logits[0].topk(5).indices])


if __name__ == "__main__":
    main()

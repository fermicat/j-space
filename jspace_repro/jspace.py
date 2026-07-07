# j-space identification (paper Step 4): decompose a residual-stream
# activation into a sparse nonnegative combination of J-lens vectors.
#
# The J-lens vector for token t at layer l is row t of (W_U @ J_l) — the
# direction in layer-l residual space whose amplification most raises the
# model's disposition to say t in the future. An activation's j-space
# content is the nonnegative sparse code over these vectors (k <= 25 in the
# paper); the unexplained remainder is non-verbalizable "workspace-external"
# activity. The paper reports j-space capturing <10% of activation variance.
#
# Full-vocab dictionaries (151k x d_model per layer) are too heavy for a
# laptop, so we restrict the dictionary to the top-M candidates by lens
# score at the position being decomposed — the same "top entries" shortcut
# the paper uses for readout interpretation.

from dataclasses import dataclass

import torch


@dataclass
class JSpaceCode:
    """Sparse nonnegative code of one activation over J-lens vectors."""

    token_ids: list[int]  # active atoms, in selection order
    coefficients: list[float]  # nonnegative weights, same order
    explained_variance: float  # 1 - ||residual||^2 / ||h||^2


@torch.no_grad()
def jlens_dictionary(model, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Unit-normalized unembedding rows W_U[t] for `candidate_ids`: [M, d_model].

    Deviation from the paper's formulation (documented in the plan's living
    notes): the paper decomposes the raw activation h onto rows of (W_U J_l).
    On this 0.8B model that decomposition is dominated by h's non-verbalizable
    bulk and yields junk atoms. We instead decompose the *transported* vector
    J_l h onto unembedding directions — the same sparse-nonnegative-code idea
    expressed in the final (verbalizable) basis, which is what the readout
    scores actually measure. Explained variance is then the verbalizable share
    of the transported vector, not of the raw activation.
    """
    W_U = model._lm_head.weight.detach().float()  # [vocab, d_model]
    atoms = W_U[candidate_ids.to(W_U.device)]
    return atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def nonneg_pursuit(
    h: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25, tol: float = 1e-4
) -> tuple[list[int], torch.Tensor, float]:
    """Nonnegative OMP: greedily select atoms, refit coefficients by NNLS
    (lstsq + drop-negatives), stop at k atoms or when no atom correlates
    positively with the residual.

    Returns (atom_indices, coefficients, explained_variance).
    """
    h = h.float()
    device = h.device
    dictionary = dictionary.to(device)
    residual = h.clone()
    h_sq = float(h @ h)
    active: list[int] = []
    coeff = torch.zeros(0, device=device)

    for _ in range(k):
        corr = dictionary @ residual  # [M]
        if active:
            corr[torch.tensor(active, device=device)] = float("-inf")
        best = int(corr.argmax())
        if corr[best] <= tol * h.norm():
            break
        active.append(best)

        # NNLS on the active set: solve unconstrained, drop negative atoms.
        while True:
            D = dictionary[torch.tensor(active, device=device)]  # [a, d]
            sol = torch.linalg.lstsq(D.T, h.unsqueeze(1)).solution.squeeze(1)
            if (sol >= 0).all() or len(active) == 1:
                coeff = sol.clamp_min(0.0)
                break
            keep = [i for i, c in zip(active, sol.tolist()) if c > 0]
            if not keep:
                coeff = sol.clamp_min(0.0)
                break
            active = keep
        residual = h - coeff @ dictionary[torch.tensor(active, device=device)]

    explained = 1.0 - float(residual @ residual) / max(h_sq, 1e-12)
    return active, coeff, explained


@torch.no_grad()
def decompose_position(
    model,
    lens,
    activations: torch.Tensor,  # [d_model] residual at (layer, position)
    lens_logits: torch.Tensor,  # [vocab] lens logits at same point
    layer: int,
    *,
    n_candidates: int = 2000,
    k: int = 25,
    display_mask: torch.Tensor | None = None,
) -> JSpaceCode:
    """j-space code of one activation, dictionary restricted to the lens's
    top `n_candidates` tokens (optionally masked to word-like tokens)."""
    scores = lens_logits.clone()
    if display_mask is not None:
        scores[~display_mask.to(scores.device)] = float("-inf")
    candidate_ids = scores.topk(n_candidates).indices.cpu()
    # Pursuit runs on CPU: the active-set solves are tiny, and MPS lacks
    # linalg_lstsq. Only the dictionary construction stays on the model device.
    dictionary = jlens_dictionary(model, candidate_ids).cpu()
    transported = lens.transport(activations.float(), layer)
    idx, coeff, explained = nonneg_pursuit(transported.cpu(), dictionary, k=k)
    order = torch.argsort(coeff, descending=True)
    return JSpaceCode(
        token_ids=[int(candidate_ids[idx[i]]) for i in order],
        coefficients=[float(coeff[i]) for i in order],
        explained_variance=explained,
    )

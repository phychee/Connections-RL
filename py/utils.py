import torch

def get_all_combos(n: int, k:int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    # this returns combinations in sorted order
    return torch.combinations(idx, r=k)

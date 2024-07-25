from typing import List

import torch


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)

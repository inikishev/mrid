import random
from collections.abc import Sequence, Callable
from typing import Any, Literal, TypeVar

ARR = TypeVar("ARR", bound=Any)

def crop(
    arr: ARR,
    reduction: Sequence[int],
    where: Literal["start", "end", "center", "random"] = "center",
) -> ARR:
    """Crop ``arr`` such that ``output.shape[i] = input.shape[i] - reduction[i]``"""

    shape = arr.shape[-len(reduction):]
    slices = []

    for r, sh in zip(reduction, shape):
        if r == 0:
            slices.append(slice(None))
            continue

        if r < 0: raise ValueError(f"Reduction {r} cannot be negative")
        if r > sh: raise ValueError(f"Reduction {r} exceeds dimension size {sh}")

        if where == 'start': start, end = 0, sh - r
        elif where == 'end': start, end = r, sh
        elif where == 'center':
            start = r // 2
            end = start + (sh - r)
        elif where == 'random':
            start = random.randint(0, r)
            end = start + (sh - r)
        else:
            raise ValueError(f"Invalid where: {where}")

        slices.append(slice(start, end))

    # apply with broadcasting
    return arr[(..., *slices)]

def crop_to_shape(
    input: ARR,
    shape: Sequence[int],
    where: Literal["start", "end", "center", "random"] = "center",
) -> ARR:
    """Crop ``input`` to ``shape``."""

    # broadcast
    if len(shape) < input.ndim:
        shape = list(input.shape[:input.ndim - len(shape)]) + list(shape)

    return crop(input, [i - j for i, j in zip(input.shape, shape)], where=where)

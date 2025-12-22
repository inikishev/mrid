import random
from collections.abc import Sequence, Callable
from typing import Any, Literal, TypeVar

import numpy as np
import torch
from . import cropping
from ..utils.python_utils import reduce_dim

ARR = TypeVar("ARR", bound=Any)

@torch.no_grad
def pad(
    input: ARR,
    padding: Sequence[int],
    mode: str = "constant",
    value = None,
    where: Literal["center", "start", "end"] = "center",
    crop: bool = False,
) -> ARR:
    """
    TODO REFACTOR

    ``output.shape[i] = input.shape[i] + padding[i]``.

    Args:
        input (torch.Tensor): input to pad.
        padding (str): how much padding to add per each dimension of ``input``.
        mode (str, optional):
            padding mode (https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
            Defaults to 'constant'.
        value (_type_, optional): padding constant value. Defaults to None.
        where (str, optional): where to pad.
            if ``center``, will pad start and end of each dimension evenly,
            if ``start``, will pad at the start of each dimension,
            if ``end``, will pad at the end. Defaults to 'center'.
        crop (bool, optional): allow cropping if padding is negative. Defaults to False.

    Returns:
        torch.Tensor: Padded `input`.
    """
    pad_values = [i if i > 0 else 0 for i in padding]

    if sum(pad_values) > 0:

        # create padding sequence for torch.nn.functional.pad
        if where == 'center':
            dims_padding = [(int(i / 2), int(i / 2)) if i % 2 == 0 else (int(i / 2), int(i / 2) + 1) for i in padding]
        elif where == 'start':
            dims_padding = [(i, 0) for i in padding]
        elif where == 'end':
            dims_padding = [(0, i) for i in padding]
        else: raise ValueError(f'Invalid where: {where}')

        # broadcasting (e.g. if padding 3×128×128 by [4, 4], it will pad by [0, 4, 4])
        if len(dims_padding) < input.ndim:
            dims_padding = [(0, 0)] * (input.ndim - len(dims_padding)) + dims_padding

        if mode == 'zeros': mode = 'constant'; value = 0
        elif mode == 'min': mode = 'constant'; value = float(input.min())
        elif mode == 'max': mode = 'constant'; value = float(input.max())
        elif mode == 'mean': mode = 'constant'; value = float(input.mean())

        if isinstance(input, np.ndarray):
            if mode == 'constant': kwargs = {"constant_values": value}
            else: kwargs = {}
            input = np.pad(input, pad_width = dims_padding, mode = mode, **kwargs) # type:ignore
        else:
            input = torch.nn.functional.pad(input, reduce_dim(reversed(dims_padding)), mode=mode, value=value) # type:ignore

    if crop:
        crop_values = [-i if i < 0 else 0 for i in padding]
        if sum(crop_values) > 0:
            input =  cropping.crop(input, crop_values, where=where)

    return input

def pad_to_shape(
    input:ARR,
    shape:Sequence[int],
    mode:str = "constant",
    value=None,
    where:Literal["center", "start", "end"] = "center",
    crop = False,
) -> ARR:
    # broadcasting
    if len(shape) < input.ndim:
        shape = list(input.shape[:input.ndim - len(shape)]) + list(shape)

    return pad(
        input=input,
        padding=[shape[i] - input.shape[i] for i in range(input.ndim)],
        mode=mode,
        value=value,
        where=where,
        crop = crop,
    )

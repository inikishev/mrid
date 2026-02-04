from collections.abc import Callable

import numpy as np
import SimpleITK as sitk

from ..loading import ImageLike, tositk


def sitk_apply_numpy(image: ImageLike, func: Callable[[np.ndarray], np.ndarray]) -> sitk.Image:
    image = tositk(image)
    array = sitk.GetArrayFromImage(image)
    shape = array.shape
    array = func(array)
    if shape != array.shape:
        raise RuntimeError(f"Function {func} changed array shape from {shape} to {array.shape}.")
    res = sitk.GetImageFromArray(array)
    res.CopyInformation(image)
    return res
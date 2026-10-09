from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk

from ..loading.convert import tositk, ImageLike


def resample_to(input: ImageLike, to: ImageLike, interpolation=sitk.sitkNearestNeighbor) -> sitk.Image:
    """Resample ``input`` to ``reference``.

    Resampling uses spatial information embedded in the sitk.Image - size, origin, spacing and direction.

    Note that this information is only available when certain imaging formats are loaded, such as DICOM and NIfTI.

    ``input`` is transformed in such a way that those attributes will match ``reference``.
    """
    return sitk.Resample(tositk(input), tositk(to), sitk.Transform(), interpolation)


def resize(img: ImageLike, new_size: Sequence[int], interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resize ``sitk.Image`` to ``new_size`` (numpy axis order, i.e. reversed of sitk size).
    Retains correct spatial information: origin and direction are preserved and spacing is
    scaled so the physical field of view stays constant."""
    img = tositk(img)
    new_size = list(reversed(new_size))

    old_size = img.GetSize()
    old_spacing = img.GetSpacing()
    new_spacing = []
    for n, o, s in zip(new_size, old_size, old_spacing):
        if n > 1:
            new_spacing.append((o - 1) * s / (n - 1))
        else:
            new_spacing.append(s)

    reference_image = sitk.Image(new_size, img.GetPixelID())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetSpacing(new_spacing)
    reference_image.SetDirection(img.GetDirection())

    return sitk.Resample(img, reference_image, sitk.Transform(), interpolator, 0.0)

def downsample(image:ImageLike, factor:float, dims: int | Sequence[int] | None, interpolator=sitk.sitkLinear) -> sitk.Image:
    """factor = 2 for 2x downsampling"""
    if isinstance(dims, int): dims = (dims, )
    image = tositk(image)
    size = sitk.GetArrayFromImage(image).shape
    size = [round(s/factor) if (dims is None or i in dims) else s for i,s in enumerate(size)]
    return resize(image, size, interpolator=interpolator)
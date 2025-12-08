import SimpleITK as sitk

from ..loading.convert import ImageLike, tositk


def expand_binary_mask(binary_mask: ImageLike, expand: int) -> sitk.Image:
    """Expand or dilate a binary mask.

    Args:
        binary_mask (ImageLike): mask
        expand (int, optional):
            Positive values expand the mask by this many pixels;
            Negative values dilate the mask by this many pixels.
    """
    binary_mask = tositk(binary_mask)
    if expand > 0:
        inverted_mask = 1 - binary_mask
        return 1 - sitk.BinaryDilate(inverted_mask, (expand, expand, expand))

    if expand < 0:
        return sitk.BinaryDilate(binary_mask, (-expand, -expand, -expand))

    return binary_mask
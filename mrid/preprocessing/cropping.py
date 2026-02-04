from collections.abc import Mapping, Sequence
from typing import Any
import SimpleITK as sitk

from ..loading.convert import tositk, ImageLike

def _get_bbox(image: sitk.Image):
    rescaled = sitk.RescaleIntensity(image, 0, 255)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(sitk.OtsuThreshold(rescaled, 0, 255))
    return filt.GetBoundingBox(255)


def crop_bg(image: ImageLike) -> sitk.Image:
    """Crops black background of a single 3D image via Otsu's thresholding.

    Args:
        image (ImageLike): Input 3D image to be cropped. Can be any format supported by tositk conversion.

    Returns:
        sitk.Image: Cropped image with black background removed, maintaining the same pixel type as input.
    """
    image = tositk(image)
    bbox = _get_bbox(image)
    return sitk.RegionOfInterest( image, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)],)

def crop_bg_D(images: Mapping[str, ImageLike], key: str) -> dict[str, sitk.Image]:
    """Finds the bounding box of ``images[key]`` and crops all images in ``images`` to that bounding box."""
    images = {k: tositk(v) for k,v in images.items()}
    reference = images[key]

    bbox = _get_bbox(reference)

    ret = {k: sitk.RegionOfInterest(v, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)]) for k,v in images.items()}
    return ret

def center_crop_or_pad(image, size: Sequence[int]) -> "sitk.Image":#[192, 224, 192]
    image = tositk(image)
    current_size = list(image.GetSize())

    low_pad = []
    high_pad = []
    low_crop = []
    high_crop = []

    for cur, tar in zip(current_size, size):
        diff = tar - cur
        if diff >= 0:
            low = diff // 2
            high = diff - low
            low_pad.append(low)
            high_pad.append(high)
            low_crop.append(0)
            high_crop.append(0)
        else:
            diff = abs(diff)
            low = diff // 2
            high = diff - low
            low_pad.append(0)
            high_pad.append(0)
            low_crop.append(low)
            high_crop.append(high)

    image = sitk.ConstantPad(image, low_pad, high_pad, 0)
    image = sitk.Crop(image, low_crop, high_crop)

    # Verify size
    if list(image.GetSize()) == size:
        return image

    raise RuntimeError(
        f"Error: Final size is {image.GetSize()}, instead of {size} (basically crop_or_pad failed for some reason)")

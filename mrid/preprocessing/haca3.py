""""
This requires [HACA3](https://github.com/lianruizuo/haca3).

HACA3 can be installed from source or through singularity image. Mrid currently only supports it being installed in a separate environment (because thats how I currently need to use it), let me know if you need other installation methods supported.

You also need to download HACA3 weights harmonization.pt and fusion model weights fusion.pt from the ``4. Usage: Inference`` section in HACA3 github readme.
"""
import os
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import SimpleITK as sitk

from ..loading import ImageLike, tositk
from .cropping import center_crop_or_pad
from .simple_elastix import register, register_D

# - ```--in-path```: file path to input source image. Multiple ```--in-path``` may be provided if there are multiple
# source images. See the above example for more details.
# - ```--target-image```: file path to target image. HACA3 will match the contrast of source images to this target image.
# - ```--target-theta```: In HACA3, ```theta```
# is a two-dimensional representation of image contrast. Target image contrast can be directly specified by providing
# a ```theta``` value, e.g., ```--target-theta 0.5 0.5```. Note: either ```--target-image``` or ```--target-image``` must
# be provided during inference. If both are provided, only ```--target-theta``` will be used.
# - ```--norm-val```: normalization value.
# - ```--out-path```: file path to harmonized image.
# - ```--harmonization-model```: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP data can
# be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
# - ```--fusion-model```: pretrained fusion model weights. HACA3 uses a 3D convolutional network to combine multi-orientation
# 2D slices into a single 3D volume. Pretrained fusion model can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
# - ```--save-intermediate```: if specified, intermediate results will be saved. Default: ```False```. Action: ```store_true```.
# - ```--intermediate-out-dir```: directory to save intermediate results.
# - ```--gpu-id```: integer number specifies which GPU to run HACA3.
# - ```--num-batches```: During inference, HACA3 takes entire 3D MRI volumes as input. This may cause a considerable amount
# GPU memory. For reduced GPU memory consumption, source images maybe divided into smaller batches.
# However, this may slightly increase the inference time.


def run_HACA3(
    conda_path: str | os.PathLike,
    env_name: str,
    harmonization_model: str | os.PathLike,
    fusion_model: str | os.PathLike,
    in_path: str | os.PathLike | Sequence[str | os.PathLike],
    out_path: str | os.PathLike,
    target_image: str | os.PathLike | None,
    target_theta: tuple[float,float] | None,
    norm_val: float | None = None,
    intermediate_out_dir: str | os.PathLike | None = None,
    gpu_id: int | None = None,
    num_batches: int | None = None,
) -> None:
    """Runs ``HACA3`` command-line routine via ``subprocess.run``.

    Args:
        conda_path: path to ``minconda3`` directory.
        env_name: name of the conda env where HACA3 is installed.
        harmonization_model: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP
            data can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
        fusion_model: pretrained fusion model weights. HACA3 uses a 3D convolutional network to
            combine multi-orientation 2D slices into a single 3D volume. Pretrained fusion model
            can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
        in_path: file path to input source image. Multiple paths may be provided if there are
            multiple source images (different modalities). Note that modalities must be in
            MNI space (1mm isotropic resolution). HACA3 assumes a spatial dimension of 192x224x192.
        out_path: file path to harmonized image.
        target_image: file path to target image. HACA3 will match the contrast of
            source images to this target image.
        target_theta: In HACA3, ```theta``` is a two-dimensional representation of image contrast.
            Target image contrast can be directly specified by providing a ```theta``` value, e.g.,
            ```target_theta = (0.5, 0.5)```. Note: either ```target_image``` or ```target_theta```
            must be provided during inference.
        norm_val: normalization value. Defaults to None.
        intermediate_out_dir: directory to save intermediate results. Defaults to None.
        gpu_id: integer number specifies which GPU to run HACA3. Defaults to None.
        num_batches: During inference, HACA3 takes entire 3D MRI volumes as input.
            This may cause a considerable amount GPU memory. For reduced GPU memory consumption,
            source images maybe divided into smaller batches. However, this may slightly
            increase the inference time. Defaults to None.

    """
    # validate env name
    env_name = str(env_name)
    if not all((c.isalnum() or c in ("-_")) for c in env_name):
        raise RuntimeError(f"env_name contains invalid characters: {env_name}")

    haca3_command = ['haca3-test']

    if isinstance(in_path, (str, os.PathLike)):
        in_path = [in_path]

    if target_theta is not None:
        if not isinstance(target_theta, tuple):
            raise RuntimeError(f"target_theta must be tuple of two float values or None, got {type(target_theta)}")
        if not len(target_theta) == 2:
            raise RuntimeError(f"target_theta must be a length 2 tuple, got length {len(target_theta)}")
        if not all(isinstance(t, (int,float)) for t in target_theta):
            raise RuntimeError(
                f"target_theta must be tuple of two float values, got tuple({tuple(type(v) for v in target_theta)})")

    for f in in_path:
        haca3_command.append(f'--in-path "{os.path.normpath(f)}"',)

    haca3_command.extend([
        f'--out-path "{os.path.normpath(out_path)}"',
        f'--harmonization-model "{os.path.normpath(harmonization_model)}"',
        f'--fusion-model "{os.path.normpath(fusion_model)}"',
    ])

    if target_image is not None: haca3_command.append(f'--target-image "{os.path.normpath(target_image)}"')
    if target_theta is not None: haca3_command.append(f'--target-theta {float(target_theta[0])} {float(target_theta[1])}')
    if norm_val is not None: haca3_command.append(f'--norm-val {float(norm_val)}')
    if intermediate_out_dir is not None:
        haca3_command.append('--save-intermediate')
        haca3_command.append(f'--intermediate-out-dir "{os.path.normpath(intermediate_out_dir)}"')
    if gpu_id is not None: haca3_command.append(f'--gpu-id {int(gpu_id)}')
    if num_batches is not None: haca3_command.append(f'--num-batches {int(num_batches)}')

    # print(f". {conda_path}/etc/profile.d/conda.sh && conda activate {env_name} && {' '.join(haca3_command)}")

    # so
    # shlex.split doesn't work on ., and conda run doesn't work for whatever reason, so we have to do this
    # all args are explicitly validated so should be ok
    command = f". {conda_path}/etc/profile.d/conda.sh && conda activate {env_name} && {' '.join(haca3_command)}"

    # run
    subprocess.run(command, shell=True, check=True)

def harmonize(
    conda_path: str | os.PathLike,
    env_name: str,
    harmonization_model: str | os.PathLike,
    fusion_model: str | os.PathLike,
    inputs: "ImageLike | Sequence[ImageLike]",
    target_image: "ImageLike | None" = None,
    target_theta: tuple[float,float] | None = None,
    norm_val: float | None = None,
    intermediate_out_dir: str | os.PathLike | None = None,
    gpu_id: int | None = None,
    num_batches: int | None = None,
) -> sitk.Image:
    """Harmonizes ``inputs`` using HACA3.

    Important: Some preprocessing steps are needed before running HACA3:

    - Inhomogeneity correction
    - Super-resolution for 2D acquired scans. This step is optional, but recommended for optimal performance. See [SMORE](https://github.com/volcanofly/SMORE-Super-resolution-for-3D-medical-images-MRI) for more details.
    - Registration to MNI space (1mm isotropic resolution). HACA3 assumes a spatial dimension of 192x224x192.

    You can inhomogeneity correction via ``mrid.n4_bias_field_correction(...)``,
    register to MNI152 via ``mrid.simple_elastix.register(...)``, and center-pad to required dimension iva
    ``mrid.center_crop_or_pad(image, [192, 224, 192])``.

    Args:
        conda_path: path to ``minconda3`` directory.
        env_name: name of the conda env where HACA3 is installed.
        harmonization_model: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP
            data can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
        fusion_model: pretrained fusion model weights. HACA3 uses a 3D convolutional network to
            combine multi-orientation 2D slices into a single 3D volume. Pretrained fusion model
            can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
        inputs: input source images. Multiple images may be provided if there are
            multiple source images (different modalities). Note that modalities must be in
            MNI space (1mm isotropic resolution). HACA3 assumes a spatial dimension of 192x224x192.
        target_image: target image. HACA3 will match the contrast of
            source images to this target image. One of ``target_image`` or ``target_theta`` must be set.
        target_theta: In HACA3, ```theta``` is a two-dimensional representation of image contrast.
            Target image contrast can be directly specified by providing a ```theta``` value, e.g.,
            ```target_theta = (0.5, 0.5)```.  One of ``target_image`` or ``target_theta`` must be set.
            Some example values: ``(10.0, 20.0)`` or ``(0.0, 30.0)`` represent T1w;
            ``(-18.0, -16.0)`` represents T2W; ``(0.0, 0.0)`` represents FLAIR.
        norm_val: normalization value. Defaults to None.
        intermediate_out_dir: directory to save intermediate results. Defaults to None.
        gpu_id: integer number specifies which GPU to run HACA3. Defaults to None.
        num_batches: During inference, HACA3 takes entire 3D MRI volumes as input.
            This may cause a considerable amount GPU memory. For reduced GPU memory consumption,
            source images maybe divided into smaller batches. However, this may slightly
            increase the inference time. Defaults to None.

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        _description_
    """

    if all(i is None for i in [target_image, target_theta]):
        raise RuntimeError("Either target_image or target_theta must be set")

    if all(i is not None for i in [target_image, target_theta]):
        raise RuntimeError("Only one of target_image or target_theta must be set")

    if isinstance(inputs, (str, os.PathLike)) or not isinstance(inputs, Sequence):
        inputs = (inputs, )

    inputs = [tositk(img) for img in inputs]
    if target_image is not None: target_image = tositk(target_image)

    # --------------------------------- run HACA3 -------------------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(inputs):
            if tuple(img.GetSize()) != (192, 224, 192):
                raise RuntimeError(
                    "all inputs must be in MNI152 space and center-padded to size of ``[192, 224, 192]``. "
                    f"Got image {i} of size {img.GetSize()}")

            sitk.WriteImage(img, os.path.join(tmpdir, f"input_{i}.nii.gz"))

        if target_image is not None:
            target_path = os.path.join(tmpdir, "target_image.nii.gz")
            if tuple(target_image.GetSize()) != (192, 224, 192):
                raise RuntimeError(
                    "all inputs must be in MNI152 space and center-padded to size of ``[192, 224, 192]``. "
                    f"Got target_image of size {target_image.GetSize()}")

            sitk.WriteImage(target_image, target_path)
        else:
            target_path = None

        run_HACA3(
            conda_path=conda_path,
            env_name=env_name,
            harmonization_model=harmonization_model,
            fusion_model=fusion_model,
            in_path = [os.path.join(tmpdir, f"input_{i}.nii.gz") for i in range(len(inputs))],
            out_path = os.path.join(tmpdir, "output.nii.gz"),
            target_image = target_path,
            target_theta = target_theta,
            norm_val = norm_val,
            intermediate_out_dir = intermediate_out_dir,
            gpu_id = gpu_id,
            num_batches = num_batches,
        )

        harmonized = tositk(os.path.join(tmpdir, "output_harmonized_fusion.nii.gz"))

    return harmonized

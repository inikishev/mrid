""""
This requires [HACA3](https://github.com/lianruizuo/haca3).

HACA3 can be installed from source or through singularity image. Mrid currently only supports it being installed in a separate environment (because thats how I currently need to use it), let me know if you need other installation methods supported.

You also need to download HACA3 weights harmonization.pt and fusion model weights fusion.pt from the ``4. Usage: Inference`` section in HACA3 github readme.
"""
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..loading import tositk

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
    in_path: str | os.PathLike | Sequence[str | os.PathLike],
    out_path: str | os.PathLike,
    target_image: str | os.PathLike | None,
    target_theta: tuple[float,float] | None,
    harmonization_model: str | os.PathLike,
    fusion_model: str | os.PathLike,
    norm_val: float | None = None,
    intermediate_out_dir: str | os.PathLike | None = None,
    gpu_id: int | None = None,
    num_batches: int | None = None,
) -> None:
    """Runs ``HACA3`` command-line routine via ``subprocess.run``.

    Note that turns out I don't have enough VRAM to run HACA3. So I haven't tested this.

    Args:
        conda_path: path to ``minconda3`` directory.
        env_name: name of the conda env where HACA3 is installed.
        in_path: file path to input source image. Multiple paths may be provided if there are multiple source images (different modalities). Note that modalities must be in MNI space (1mm isotropic resolution). HACA3 assumes a spatial dimension of 192x224x192.
        out_path: file path to harmonized image.
        target_image: file path to target image. HACA3 will match the contrast of source images to this target image.
        target_theta: In HACA3, ```theta``` is a two-dimensional representation of image contrast. Target image contrast can be directly specified by providing a ```theta``` value, e.g., ```target_theta = (0.5, 0.5)```. Note: either ```target_image``` or ```target_theta``` must be provided during inference.
        harmonization_model: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP data can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
        fusion_model: pretrained fusion model weights. HACA3 uses a 3D convolutional network to combine multi-orientation
2D slices into a single 3D volume. Pretrained fusion model can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
        norm_val: normalization value. Defaults to None.
        intermediate_out_dir: directory to save intermediate results. Defaults to None.
        gpu_id: _description_. Defaults to None.
        num_batches: During inference, HACA3 takes entire 3D MRI volumes as input. This may cause a considerable amount
GPU memory. For reduced GPU memory consumption, source images maybe divided into smaller batches. However, this may slightly increase the inference time.. Defaults to None.

    """
    # validate env name
    env_name = str(env_name)
    if not all((c.isalnum() or c in ("-_")) for c in env_name):
        raise RuntimeError(f"env_name contains invalid characters: {env_name}")

    haca3_command = ['haca3-test']

    if isinstance(in_path, (str, os.PathLike)):
        in_path = [in_path]

    if target_theta is not None:
        assert isinstance(target_theta, tuple)
        assert len(target_theta) == 2
        assert all(isinstance(t, (int,float)) for t in target_theta)

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
    # shlex.split doesn't work on ., and conda run doesn't work, so we have to do this
    command = f". {conda_path}/etc/profile.d/conda.sh && conda activate {env_name} && {' '.join(haca3_command)}"

    # run
    subprocess.run(command, shell=True, check=True)


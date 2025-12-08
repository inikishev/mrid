"""
To install synthstrip, go to https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
and find the "SynthStrip Tool" section.

If you have FreeSurfer installed, you just need to find path to the "mri_synthstrip" file.

If you do not want to install FreeSurfer, you can run SynthStrip in a container.
The webpage has two commands, one for Apptainer/Singularity, and one for Docker.
Pick one and run it, this will download a script file which installs synthstrip container
if it isn't installed, and provides a command-line interface for it.

Pass path to the script to mrid functions in this module.
"""
import os
import subprocess

# Running SynthStrip version 1.8 from Docker
# usage: mri_synthstrip [-h] -i FILE [-o FILE] [-m FILE] [-d FILE] [-g]
#                       [-b BORDER] [-t THREADS] [-f FILL] [--no-csf]
#                       [--model FILE]

# Robust, universal skull-stripping for brain images of any type.

# options:
#   -h, --help            show this help message and exit
#   -i FILE, --image FILE
#                         input image to skullstrip
#   -o FILE, --out FILE   save stripped image to file
#   -m FILE, --mask FILE  save binary brain mask to file
#   -d FILE, --sdt FILE   save distance transform to file
#   -g, --gpu             use the GPU
#   -b BORDER, --border BORDER
#                         mask border threshold in mm, defaults to 1
#   -t THREADS, --threads THREADS
#                         PyTorch CPU threads, PyTorch default if unset
#   -f FILL, --fill FILL  BG fill value, defaults to min(image.min, 0)
#   --no-csf              exclude CSF from brain border
#   --model FILE          alternative model weights

# If you use SynthStrip in your analysis, please cite:
# ----------------------------------------------------
# SynthStrip: Skull-Stripping for Any Brain Image
# A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
# NeuroImage 206 (2022), 119474
# https://doi.org/10.1016/j.neuroimage.2022.119474

# Website: https://synthstrip.io


def run_synthstrip(
    synthstrip_script_path: str | os.PathLike,
    image: str | os.PathLike,
    out: str | os.PathLike | None,
    mask: str | os.PathLike | None = None,
    sdt: str | os.PathLike | None = None,
    gpu: bool | None = None,
    border: int | None = None,
    threads: int | None = None,
    fill: int | None = None,
    no_csf: bool | None = None,
    model: str | os.PathLike | None = None,
):
    """Robust, universal skull-stripping for brain images of any type

    If you use SynthStrip in your analysis, please cite:

    SynthStrip: Skull-Stripping for Any Brain Image
    A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
    NeuroImage 206 (2022), 119474
    https://doi.org/10.1016/j.neuroimage.2022.119474

    Website: https://synthstrip.io

    Args:
        synthstrip_script_path (str | os.PathLike):
            path to synthstrip script.
        image (str | os.PathLike): input image to skullstrip.
        out (str | os.PathLike | None): save stripped image to file.
        mask (str | os.PathLike | None): save binary brain mask to file. Defaults to None.
        sdt (str | os.PathLike | None, optional): save distance transform to file. Defaults to None.
        gpu (bool | None, optional): use the GPU, defaults to False if unset.
        border (int | None, optional): mask border threshold in mm, defaults to 1 if unset.
        threads (int | None, optional): PyTorch CPU threads, PyTorch default if unset.
        fill (int | None, optional): BG fill value, defaults to min(image.min, 0) if unset.
        no_csf (bool | None, optional): exclude CSF from brain border.
        model (str | os.PathLike | None, optional): alternative model weights
    """
    command = [
        "python",
        os.path.normpath(synthstrip_script_path),
        "-i", os.path.normpath(image),
    ]

    if out is not None: command.extend(["-o", os.path.normpath(out)])
    if mask is not None: command.extend(["-m", os.path.normpath(mask)])
    if sdt is not None: command.extend(["-d", os.path.normpath(sdt)])
    if gpu is not None: command.append("-g")
    if border is not None: command.extend(["-b", f"{border}"])
    if threads is not None: command.extend(["-t", f"{threads}"])
    if fill is not None: command.extend(["-f", f"{fill}"])
    if no_csf is not None: command.append("--no_csf")
    if model is not None: command.extend(["--model", os.path.normpath(model)])

    # run dcm2niix
    subprocess.run(command, check=True)


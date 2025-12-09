""""
This requires CTseg docker image to be present in the system.

If you haven't already, install docker to your OS following this https://docs.docker.com/engine/install/

CTseg is available in the following repository: https://github.com/WCHN/CTseg

Navigate to any folder, open the terminal in that folder and type in ``git clone https://github.com/WCHN/CTseg``. You might have to [install git](https://git-scm.com/install/) if you don't have it installed. This will create a new directory called ``CTseg`` in the folder and download the repository to it. Alternatively you can open the repository in your web browser, click on the green "Code" button near the top and click "Download ZIP", and unpack the archive to a folder named ``CTseg``.

Then type in the command specified in "Docker" section of the read-me in https://github.com/WCHN/CTseg to build an image from the Dockerfile in this repository. I decided to not copy the command here in case it gets updated in the repository. This will download and build a docker image, usually to ``/var/lib/docker/`` on Linux, or inside the Docker Desktop virtual machine disk image on Windows. Note that this will download 3 GB of data, so it might take some time.

After it is done, the CTseg functions from mrid can be used.
"""
import os
import subprocess
from pathlib import Path

from ..loading import tositk


def run_CTseg(
    ct_nii_file: str | os.PathLike,
    docker_image = "ubuntu:ctseg",
) -> None:

    # docker run --rm -it -v "/home/jj/data":/data ubuntu:ctseg function spm_CTseg '/data/CT.nii'
    # better
    # docker run --rm -it -v "/home/jj/data":/data ubuntu:ctseg eval "spm_CTseg('/data/CT.nii', '', true, true, true, true, 1.0)"
    ct_nii_file = Path(ct_nii_file)

    command = [
        "docker",
        "run",

        # --rm automatically removes the container's file system after the container exits. This is useful for running temporary containers.
        "--rm",

        # This is a combination of two flags, -i and -t:
            # -i (interactive): Keeps the standard input (STDIN) open, allowing you to interact with the container.
            # -t (tty): Allocates a pseudo-TTY, which makes the container behave like a normal terminal session.
        "-it",

        # -v used for mounting volumes, which allows you to connect a file path on your host machine to a path inside the container. This option requires a specific format: -v <host_path>:<container_path>.
        "-v",
        f"{os.path.normpath(ct_nii_file.parent)}:/data",

        # docker image name
        docker_image,

        # CTseg func
        "spm_CTseg",

        # path to CT scan
        f"/data/{ct_nii_file.name}",
    ]

    # run dcm2niix
    subprocess.run(command, check=True)


a = 4
""""
If you haven't already, install docker to your OS following this https://docs.docker.com/engine/install/

CTseg is available in the following repository: https://github.com/WCHN/CTseg

Navigate to any folder, open the terminal in that folder and type in ``git clone https://github.com/WCHN/CTseg``. You might have to [install git](https://git-scm.com/install/) if you don't have it installed. This will create a new directory called ``CTseg`` in the folder and download the repository to it. Alternatively you can open the repository in your web browser, click on the green "Code" button near the top and click "Download ZIP", and unpack the archive to a folder named ``CTseg``.

Then type in the command specified in "Docker" section of the read-me in https://github.com/WCHN/CTseg to build an image from the Dockerfile in this repository. I decided to not copy the command here in case it gets updated in the repository. This will download and build a docker image, usually to ``/var/lib/docker/`` on Linux, or inside the Docker Desktop virtual machine disk image on Windows. Note that this will download 3 GB of data, so it might take some time.


"""
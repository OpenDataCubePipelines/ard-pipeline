# ARD Pipeline

A Python package for producing standarised imagery in the form of:

* Nadir Bi-directional Reflectance Distribution Function Adjusted Reflectance (NBAR)
* NBART; NBAR with Terrain Illumination correction
* Surface Brightness Temperature
* Pixel Quality (per pixel metadata)

The luigi task workflow for producing NBAR for a Landsat 5TM scene is given below.

<div style="background-color: white; display: inline-block;">
  <img src="docs/source/diagrams/luigi-task-visualiser.svg" alt="Luigi Task Visualiser">
</div>

## Supported Satellites and Sensors

* Landsat 5 TM
* Landsat 7 ETM
* Landsat 8+9 OLI+TIRS
* Sentinel 2a+b+c

## Development


A [Justfile](https://github.com/casey/just) is included in the repo for running common commands.

Build docker container:

    just build

Run a shell inside it:

    just run

Run tests:

    just test

### Dependencies

You can either create your own local Python environment, or use the provided [Dockerfile](Dockerfile).

Builds are also available from Dockerhub:

    docker pull --platform linux/amd64 geoscienceaustralia/ard-pipeline:dev

If building your own environment, Miniconda is recommended due to the large number of
native dependencies.

A script is provided to build Conda with dependencies:

```Bash
# Create environment in ~/conda directory
./deployment/create-conda-environment.sh ~/conda

# Activate the environment in the current shell
. ~/conda/bin/activate

# Ensure build dependencies are installed
python3 -m setuptools_scm  # prints version string if OK

# Install ARD for development
pip install --no-build-isolation --editable .

# Check the environment for common problems.
# (Eg, can we import dependencies on NCI?)
module use /g/data/v10/private/modules/modulefiles  # allow MODTRAN module loading
module load modtran
./deployment/check-environment.sh

# (note that the last check is for Modtran, which you may or may not be using in your environment. On NCI, we can `module load modtran`)
#
# FAILED runs will print an error, or may display Python code
#        in the terminal. This may obscure failure output.
#
# SUCCESSFUL runs will display several module imports & HDF5 tests.
```

### Import errors

If you try running code directly from the source repository, such
as in running tests, you may see import errors such as this:
`from wagl.__sat_sol_angles import angle`

These are due to the native modules (c, fortran) needing built
in the repo.

You can avoid this, and still maintain live editing, by
doing a non-isolated editable installation:

```Bash
python3 -m pip install --no-build-isolation --editable .
```

Meson will then auto-build the native modules as needed and
you can run directly from your source directory.

Run checks locally using the `./check-code.sh` file.

**setuptools_scm dependency**

The `./check-code.sh` script can fail like:

```Bash
$ ./deployment/check-environment.sh
Checking environment...
Trying rasterio... ✅ 1.3.9
Trying luigi... ✅ 3.5.0
Trying wagl... x
        No module named 'wagl._version'
Attempting load of fortran-based modules... Traceback (most recent call last):
  File "<stdin>", line 29, in <module>
  File "/g/data/u46/users/bpd578/projects/ard-pipeline/wagl/__init__.py", line 5, in <module>
    from ._version import __version__
ModuleNotFoundError: No module named 'wagl._version'
```

This indicates the `setuptools_scm` dependency is too _new_. Check the installed version with:

```Bash
pip freeze | grep setuptools_scm
```

If installed, this will display output like `setuptools-scm==8.1.0`.

The current workaround is to check the desired version of `setuptools-scm` from the `pyproject.toml` configuration. If the installed version is higher than the `pyproject.toml` version, uninstall the new version & install an older version, e.g.:

```Bash
pip install "setuptools_scm[toml]>=6.2,<8"
```

### Additional HDF5 compression filters (optional)

Additional compression filters can be used via HDF5's
[dynamically loaded filters](https://support.hdfgroup.org/archive/support/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf).
Essentially the filter needs to be compiled against the HDF5 library, and
installed into HDF5's plugin path, or a path of your choosing, and set the
HDF5_PLUGIN_PATH environment variable. The filters are then automatically
accessible by HDF5 via the [integer code](https://support.hdfgroup.org/services/contributions.html)
assigned to the filter.

#### Mafisc compression filter

Mafisc combines both a bitshuffling filter and lzma compression filter in order
to get the best compression possible at the cost of lower compression speeds.
To install the `mafisc` compression filter, follow these [instructions](https://wr.informatik.uni-hamburg.de/research/projects/icomex/mafisc).

#### Bitshuffle

The [bitshuffle filter](https://github.com/kiyo-masui/bitshuffle) can be installed
from source, or conda via the supplied [conda recipe](https://github.com/kiyo-masui/bitshuffle/tree/master/conda-recipe).
It utilises a bitshuffling filter on top of either a lz4 or lzf compression filter.


# Digital Earth Antarctica's ARD Pipeline on AWS

This section explains how to setup and run the ARD pipeline on an AWS EC2 instance running Ubuntu 22 and higher distributions. This approach is specifically developed for GA's _DE Antarctica_ Optical ARD Pipeline, but can be easily expanded to any other region.

## Setup

There are a few prerequisites before you can run a ARD Pipeline job which are listed below:

* ARD-Pipeline Project setup
* MODTRAN setup
* ARD-Pipeline downloader setup

Follow the steps in each section to setup your environment for running the pipeline.

**Note:** This guideline assumes that you are setting up and running the ARD Pipeline on an AWS EC2 machine running Ubuntu 22 or higher distributions.

### ARD-Pipeline project setup

* Clone the ARD-pipeline repository from here [ARD Pipeline](https://github.com/OpenDataCubePipelines/ard-pipeline.git) to your home directory and check out the `dea-ant/aws-migration` branch.

You can find your home directory by running `echo $HOME`.

* Create and activate the project's conda environment by running

```bash
conda create -f deployment/environment.yaml
```
and then by
```bash
conda activate ard-pipeline
```

You need a `conda` distribution to be able to create and activate conda environment. `Miniconda` distribution is the recommended one and it will be used in this guideline.

* Run the following to install additional pipeline dependencies:
```bash
pip install --no-binary :all: \
  "git+https://github.com/ubarsc/rios@rios-${rios_version:-1.4.10}#egg=rios" \
  "git+https://github.com/ubarsc/python-fmask@pythonfmask-${fmask_version:-0.5.7}#egg=python-fmask" \
  awscli boto boto3
```

* Install WAGL using the command:
```bash
pip install --no-build-isolation --editable .
```

* *OPTIONAL*: If you want to contribute to the code base, please install the `pre-commit` library with `pip install pre-commit`

### MODTRAN setup

This section assumes users have access to various GA's S3 buckets on AWS.

The pipeline requires [MODTRAN Software](http://modtran.spectral.com/modtran_index) for processing atmospheric information. To sut up a working copy of MODTRAN follow the steps below:

* Copy MODTRAN 6 to your home directory from [AWS S3](s3://imam-dev-bucket/LS9-isolated-pipeline/volumes/ancillary/MODTRAN6.0.2.3G/) by running the command below (assuming that you are in your home directory & have AWS CLI installed):

```bash
aws s3 cp s3://imam-dev-bucket/LS9-isolated-pipeline/volumes/ancillary/MODTRAN6.0.2.3G/ MODTRAN6/ --recursive
```
The total downloaded size of the data is about 30GB, so depending on your network connection speed, it might take some time to download the data.

* Add MODTRAN to the path and give it the right permissions.

```bash
export PATH="$HOME/MODTRAN6/bin/linux:$PATH"
sudo chmod -R a+x $HOME/MODTRAN6
```

* Activate MODTRAN using your license key

```bash
$HOME/MODTRAN6/bin/linux/mod6c_cons -activate_license <MODTRAN_PRODUCT_KEY>
```

You can also check your MODTRAN Software version by running

```bash
$HOME/MODTRAN6/bin/linux/mod6c_cons -version
```

which should show an output similar to this (depending n your software version): `MODTRAN(R) 6.0.2.3G`

### Checking the environment
At this stage you can check if the environment is configured correctly by running `./deployment/check-environment.sh` while you are in your ard-pipeline conda environment.

You should be able to see results similar to this:

```bash
Checking environment...
Trying rasterio... ✅ 1.3.9
Trying luigi... ✅ 3.5.0
Trying wagl... ✅ 6.1.2.dev244+gf001cc1f.d20260529
Attempting load of fortran-based modules... ✅
Attempting hdf5 blosc compression...✅
Checking modtran is available...✅
```

### ARD Pipeline Ancillary Downloader setup
To run the ARD pipeline on AWS, you will need to use the `ard-pipeline-downloader` utility, designed to access and download the required ancillary files from various sources. For the Digital Earth Antarctica pipeline, this includes BRDF, ERA5 and MERRA2 data files. You need to clone this repo [ARD Pipeline Downloader](https://github.com/arcisad/ard-pipeline-downloader.git) to your **home** directory and create the conda environment for it using the `deployment/environment.yaml` file provided in its root directory. You don't need to activate its conda environment for running the ARD pipeline.

## Running a job
You can run a job for a given scene using the provided `job.sh` file inside the `deployment/templates` directory. Create a new directory for your job (e.g. `batch_x` where x is your job number) and copy the `job.sh` file from the `deployment/templates` to your job directory.

### .env file
The ancillary downloader and the ard pipeline itself need to authenticate with various data provider platforms in order to access and download the data. You need a `.env` file in the root of your ard pipeline project directory which provides the credentials for those authentication tasks. The `.env` file should contain the following:

```bash
AWS_ACCESS_KEY_ID=<your-AWS-access-key-id>
AWS_REGION="ap-southeast-2"
AWS_SECRET_ACCESS_KEY=<your-AWS-secret-access-key>
AWS_SESSION_TOKEN=<your-AWS-session-token>
CDS_ERA5_KEY=<your-CDS-key>
CDS_ERA5_URL="https://cds.climate.copernicus.eu/api"
EARTHDATA_PASSWORD=<your-EarthData-password>
EARTHDATA_USERNAME=<your-EarthData-username>
```

Running a job for a given scene requires the `tarfile` for that scene to be present locally on the instance.
With that you can start a job by entering into your job directory and running:

```bash
./job.sh --scene-file <path-to-your-scene-tarfile>
```
An example run command is: `./job.sh --scene-file LC08_L1GT_124109_20241211_20241218_02_T2.tar`

This will automatically download the ancillary files in the downloader's conda environment, switches to the ard pipeline's environment and starts processing the scene.
Running the script will start a `Luigi` server which will keep running after the job is finished (failed or succeeded). To stop the server you can run `pkill -9 luigid`

**Note**: If you have cloned the ard pipeline repo to a directory other than your home directory, or placed the downloader project or MODTRAN directory on a path other than you home directory, you will need to pass extra input arguments to the `job.sh` command. Check the content of `job.sh` file for more information.

### Running the code inside a Docker container
The Dockerfile provided in the project directory only builds an isolated environment with necessary packages and libraries installed to run the code. It is an alternative to the local Conda environment that could be transferred to and used in other machines and platforms.

To build the Docker container run:
```bash
docker build --platform linux/amd64 -t ard:dev .
```

Then run the following to enter the container in an interactive shell mode and also mount the required volumes:

```bash
docker run --platform linux/amd64 -it --rm --volume "${PWD}:/ard-pipeline" --volume $HOME/MODTRAN6:/home/MODTRAN6 -w /ard-pipeline ard:dev /bin/bash -l
```

Run the steps below inside the container first:

```bash
export PATH="/home/MODTRAN6/bin/linux:$PATH"
pip install --no-build-isolation --editable .
```

You should be able to start jobs inside the container now.

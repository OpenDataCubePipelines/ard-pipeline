# ARD Pipeline

A Python package for producing standarised imagery in the form of:

* Nadir Bi-directional Reflectance Distribution Function Adjusted Reflectance (NBAR)
* NBART; NBAR with Terrain Illumination correction
* Surface Brightness Temperature
* Pixel Quality (per pixel metadata)

The luigi task workflow for producing NBAR for a Landsat 5TM scene is given below.

![](docs/source/diagrams/luigi-task-visualiser-reduced.png)

## Supported Satellites and Sensors

* Landsat 5 TM
* Landsat 7 ETM
* Landsat 8+9 OLI+TIRS
* Sentinel 2a+b

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
    # (Eg, can we import dependencies?)
    ./deployment/check-environment.sh

    # (note that the last check is for Modtran, which you may or may not be using in your environment. On NCI, we can `module load modtran`)

```

### Import errors

If you try running code directly from the source repository, such
as in running tests, you may see import errors such as this:
`from wagl.__sat_sol_angles import angle`

These are due to the native modules (c, fortran) needing built
in the repo.

You can avoid this, and still maintain live editing, by
doing a non-isolated editable install:

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
$ pip freeze | grep setuptools_scm
```

If installed, this will display output like `setuptools-scm==8.1.0`.

The current workaround is to check the desired version of `setuptools-scm` from the `pyproject.toml` configuration. If the installed version is higher than the `pyproject.toml` version, uninstall the new version & install an older version, e.g.:

```Bash
$ pip install "setuptools_scm[toml]>=6.2,<8"
```

### Additional HDF5 compression filters (optional)

Additional compression filters can be used via HDF5's
[dynamically loaded filters](https://support.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf).
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

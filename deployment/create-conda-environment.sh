#!/usr/bin/env bash

set -eux
unset PIP_REQUIRE_VIRTUALENV

clean_all=false
pip_args=(--no-binary :all:)

location="conda"
# If they provided a location argument, install there instead
if [ $# -gt 0 ]; then
  location="$1"
fi

mkdir -p "${location}"

# TODO: Or in Docker, should it be "$TARGETARCH"?
arch="$(uname -m)"
if [ "$arch" = "arm64" ]; then
  archname="arm64"
else
  archname="x86_64"
fi

if [ "$(uname)" = "Darwin" ]; then
  osname="MacOSX"
else
  osname="Linux"
fi

cache_dir=~/.cache
mkdir -p "${cache_dir}"
conda_file="Miniforge3-${osname}-${archname}.sh"
conda_inst="${cache_dir}/${conda_file}"

if [ ! -f "${conda_inst}" ]; then
  wget --progress=dot:giga \
    -O "${conda_inst}" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/${conda_file}"
  chmod +x "${conda_inst}"
fi

"${conda_inst}" -b -f -p "${location}"
set +ux
# dynamic, so shellcheck can't check it.
# shellcheck source=/dev/null
. "${location}/bin/activate"

conda env update -n base -f "$(dirname "$0")/environment.yaml"

# Build the HDF5 bitshuffle filter ourselves, as neither conda build is usable:
# - hdf5-external-filter-plugins-bitshuffle declares an LZ4 destination capacity of the uncompressed size, so an incompressible block is stored with a length of zero and cannot be read back
# - conda-forge's bitshuffle is compiled with -march=native, but our build host can differ from our production host (AMD vs Intel!)
bitshuffle_version=0.5.2
bitshuffle_sdist="${cache_dir}/bitshuffle-${bitshuffle_version}.tar.gz"
if [ ! -f "${bitshuffle_sdist}" ]; then
  wget --progress=dot:giga -O "${bitshuffle_sdist}" \
    "https://files.pythonhosted.org/packages/34/d3/539ae1f2c7404e5396f90a9b4cca2e0d83ed1a9c8e598f94efe88130094a/bitshuffle-${bitshuffle_version}.tar.gz"
fi
echo "dc0e3fb7bdbf42be1009cc3028744180600d625a75b31833a24aa32aeaf83d8d  ${bitshuffle_sdist}" | sha256sum -c -

# Target the base ISA: one image runs across mixed CPU generations.
march_flag=()
if [ "${archname}" = "x86_64" ]; then
  march_flag=(-march=x86-64)
fi

bitshuffle_build="$(mktemp -d)"
tar xzf "${bitshuffle_sdist}" -C "${bitshuffle_build}"
mkdir -p "${location}/lib/hdf5/plugin"
(
  cd "${bitshuffle_build}/bitshuffle-${bitshuffle_version}"
  # The same sources and flags as upstream's setup.py builds the plugin extension with.
  "${CC:-gcc}" -O3 -ffast-math -std=c99 -fPIC -shared "${march_flag[@]}" \
    -Isrc -Ilz4 -I"${location}/include" \
    src/bshuf_h5plugin.c src/bshuf_h5filter.c src/bitshuffle.c src/bitshuffle_core.c \
    src/iochain.c lz4/lz4.c \
    -L"${location}/lib" -lhdf5 \
    -o "${location}/lib/hdf5/plugin/libh5bshuf.so"
)
rm -rf "${bitshuffle_build}"

# Freeze the environment as it exists without our locally-installed  packages.
# conda env export --from-history  > "${location}/environment.yaml"

# These version defaults may be overidden by setting them before calling the script.
pip install "${pip_args[@]}" \
  "git+https://github.com/ubarsc/rios@rios-${rios_version:-1.4.10}#egg=rios" \
  "git+https://github.com/ubarsc/python-fmask@pythonfmask-${fmask_version:-0.5.7}#egg=python-fmask" \
  awscli boto boto3

if [ "$clean_all" = true ]; then
  conda clean --all -y
  rm "${conda_inst}"
fi

echo
echo "Conda installed to ${location}"
echo "Run 'source ${location}/bin/activate' to activate"

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

conda install -c conda-forge -n base mamba
conda env update -n base -f "$(dirname "$0")/environment.yaml"

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

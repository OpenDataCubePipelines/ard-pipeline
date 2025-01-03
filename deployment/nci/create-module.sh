#!/usr/bin/env bash

# ARD Pipeline Module Creation
#
# This provides a workflow to create a conda environment & install ard-pipeline.

set -eou pipefail

usage_text="
ARD Pipeline Module Creation
----------------------------

Usage:
    create-module.sh [version]

Runs a workflow to create a conda environment & install ard-pipeline, typically
on NCI systems (by default this script depends on NCI /g/data dirs & PBS).

The workflow is configurable with command line args & env variables. A single
command line argument exists 'version', which is the name of the directory
where ard-pipeline will be installed. The following environment vars can be
be overridden to customise dependencies:

module_dir: specify a path to change the install location.
swfo_version: git commit ID (a type of version)


Usage examples:
---------------

Create environment in a custom directory:
$ module_dir=/g/data/users/person/modules ./deployment/nci/create-module.sh

Create environment in a custom dir, with custom version:
$ module_dir=/g/data/users/person/modules ./deployment/nci/create-module.sh v1.0
"

# users can provide the version number as the first argument, otherwise make a date-based one
version="${1:-$(date '+%Y%m%d-%H%M')}"

# is 1st arg (the version) a help flag?
if [ "$version" == "-h" ] || [ "$version" == "--help" ]; then
  echo "$usage_text"
  exit 0
fi

# Ensure `this_path` is an absolute path to prevent create-conda-environment.sh
# from failing further in the script due to a incorrect relative path
this_dir=$(realpath "$(dirname "${0}")")
cd "${this_dir}"

full_hostname=$(hostname)

# If we're on NCI, make sure they don't have other python-containing modules loaded.
# It will interfere with the build
if [[ $full_hostname == *"nci.org.au"* ]]; then
    if [[ $(which python3) != "/bin/python3" ]]; then
        echo "'python' does not appear to be the default NCI python. Make sure you have no modules loaded."
        exit 1
    fi
fi

umask 002
unset PYTHONPATH

export LC_ALL=en_AU.utf8
export LANG=C.UTF-8

# User can set any of these bash vars before calling to override them
echo "##########################"
echo "module_dir = ${module_dir:=/g/data/v10/private/modules}"
echo
echo "swfo_version= ${swfo_version:="761dcc19cef69573ae420aec3fc3872851cc96fa"}"
echo "gost_version = ${gost_version:="gost-0.0.3"}"
echo "modtran_version = ${modtran_version:="6.0.1"}"
echo
# It thinks we're trying to quote the inner json for bash
# shellcheck disable=SC2089
echo "ard_product_array=${ard_product_array:="[\"NBART\", \"NBAR\"]"}"
echo "fmask_version=${fmask_version:="0.5.7"}"
echo
# Uppercase to match the variable that DEA modules use (If you already have it loaded, we'll take it from there).
echo "DATACUBE_CONFIG_PATH = ${DATACUBE_CONFIG_PATH:="/g/data/v10/public/modules/dea/20221025/datacube.conf"}"
echo "##########################"
export module_dir swfo_version gost_version modtran_version

echoerr() { echo "$@" 1>&2; }

#if [[ $# != 1 ]] || [[ "$1" == "--help" ]];
#then
#    echoerr
#    echoerr "Usage: $0 <tagged_ard_version>"
#    exit 1
#fi

package_name=ard-pipeline
package_description="ARD Pipeline"
package_dest=${module_dir}/${package_name}/${version}
export package_name package_description package_dest version fmask_version

# It thinks we're trying to quote the inner json for bash
# shellcheck disable=SC2090
export ard_product_array

echo
printf 'Packaging "%s %s" to "%s"\n' "$package_name" "$version" "$package_dest"
read -p "Continue? [y/N]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Proceeding..."
else
  exit 1
fi

module use /g/data/v10/public/modules/modulefiles /g/data/v10/private/modules/modulefiles
module load openmpi

echo
echo "Creating Conda environment"
export conda_root="${package_dest}/conda"
"${this_dir}/../create-conda-environment.sh" "${conda_root}"

set +u

# We cannot use `activate`, as it tries to use the caller's script parameters ("$@"), causing spurious failures.
# dynamic, so shellcheck can't check it.
# shellcheck source=/dev/null
source "${conda_root}/etc/profile.d/conda.sh"
conda activate base

# this seems to be killing conda?
# set -u

# TODO: Install from tagged version.
echo
pushd ../../
	echo "Installing ard-pipeline $(python3 -m setuptools_scm)"
	python3 -m pip install .
popd

echo
echo "Adding utility packages"
conda install -y jq

# TODO: update these? They aren't used directly by the processor.
# swfo-convert is needed for brdf downloads
python3 -m pip install \
             "git+https://github.com/OpenDataCubePipelines/swfo.git@${swfo_version}"
#             "git+https://github.com/sixy6e/mpi-structlog@develop#egg=mpi_structlog" \
#             "git+https://github.com/OpenDataCubePipelines/gost.git@${gost_version}"

echo
echo "Adding luigi configs"
mkdir -v -p "${package_dest}/etc"
envsubst < "${this_dir}/luigi.cfg.template" > "${package_dest}/etc/luigi.cfg"
cp -v "${this_dir}/luigi-logging.cfg" "${package_dest}/etc/luigi-logging.cfg"

echo
echo "Adding datacube config"
cp -v "${DATACUBE_CONFIG_PATH}" "${package_dest}/etc/datacube.conf"

echo
echo "Writing modulefile"
modulefile_dir="${module_dir}/modulefiles/${package_name}"
mkdir -v -p "${modulefile_dir}"
modulefile_dest="${modulefile_dir}/${version}"
envsubst < modulefile.template > "${modulefile_dest}"
echo "Wrote modulefile to ${modulefile_dest}"

# TODO: revoke write permissions on module?

echo
echo 'Done. Ready:'
echo "   module load ${package_name}/${version}"

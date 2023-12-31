#!/usr/bin/env bash

set -eou pipefail

this_dir="$(dirname "${0}")"
cd "${this_dir}"

umask 002
unset PYTHONPATH
module use /g/data/v10/public/modules/modulefiles /g/data/v10/private/modules/modulefiles
module load openmpi

export LC_ALL=en_AU.utf8
export LANG=C.UTF-8

# User can set any of these bash vars before calling to override them
echo "##########################"
echo "module_dir = ${module_dir:=/g/data/v10/public/modules}"
echo "swfo_version= ${swfo_version:="swfo-0.0.2"}"
echo "gost_version = ${gost_version:="gost-0.0.3"}"
echo "modtran_version = ${modtran_version:="6.0.1"}"
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

# They can provide the version number as the first argument, otherwise we'll make a date-based one.
version="${1:-$(date '+%Y%m%d-%H%M')}"

package_name=ard-pipeline
package_description="ARD Pipeline"
package_dest=${module_dir}/${package_name}/${version}
export package_name package_description package_dest version

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

echo
echo "Creating Conda environment"
export conda_root="${package_dest}/conda"
"${this_dir}/../create-conda-environment.sh" "${conda_root}"

set +u
# dynamic, so shellcheck can't check it.
# shellcheck source=/dev/null
. "${conda_root}/bin/activate"
# conda activate ard

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
# python3 -m pip install \
#             "git+https://github.com/sixy6e/mpi-structlog@develop#egg=mpi_structlog" \
#             "git+https://github.com/OpenDataCubePipelines/swfo.git@${swfo_version}" \
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

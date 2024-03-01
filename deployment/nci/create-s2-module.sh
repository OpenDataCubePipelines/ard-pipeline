#!/usr/bin/env bash

set -eou pipefail

this_dir="$(dirname "${0}")"
cd "$this_dir"

# TODO: Should we use a different module name ("ard-pipeline-s2") instead of version?
version="${1:-$(date '+%Y%m%d-%H%M')}-s2"

export ard_product_array='["NBART"]'
export fmask_version="0.5.7"

./create-module.sh "${version}"  "${@}"

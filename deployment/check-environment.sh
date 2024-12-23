#!/usr/bin/env bash

# The check environment script is intended as a fast sanity test. This ensures
# the major components of the environment are installed, can make use of wagl,
# workflow runners, HDF5 & geospatial libraries.
#
# Usage examples:
# $ check-environment.sh  # no args required

set -eu

echo Checking environment...
python_bin_path=$(which python)
if [[ ! $python_bin_path =~ "conda" ]]; then
  echo "❌ Error: python path does not contain 'conda'. Have you loaded the module?"
  exit 1
fi
python3_bin_path=$(which python3)
if [[ ! $python3_bin_path =~ "conda" ]]; then
  echo "❌ Error: python3 path does not contain 'conda'. Have you loaded the module?"
  exit 1
fi


python3 <<EOF

import sys

def bold(s:str) ->str:
    """Make bold text in the CLI, if this is a cli"""
    if sys.stdout.isatty():
        return f"\033[1m{s}\033[0m"
    else:
        return s

import importlib
def try_load(module_name:str):
    print(f'Trying {bold(module_name)}... ', end='', flush=True)
    try:
        module = importlib.import_module(module_name)
        print(f'✅ {module.__version__}')
    except ImportError as e:
        print('❌')
        print(f'\t{e.msg}')

try_load('rasterio')
try_load("luigi")
try_load("wagl")

# Does the full wagl import chain exist?
# This will import the fortran modules too, which are
# commonly missing when the build is misconfigured.
print("Attempting load of fortran-based modules... ", end='', flush=True)
from wagl import singlefile_workflow
print("✅")

# TODO CHECK: The previous import of wagl should have initialised the filters.
print("Attempting hdf5 blosc compression...", end='', flush=True)
import h5py
import tempfile
from wagl.hdf5 import H5CompressionFilter

# use wagl's custom setup for dataset creation options, which provides args
# to prevent this test from crashing
compressor = H5CompressionFilter.BLOSC_LZ
config = compressor.config(shuffle=False)
kwargs = config.dataset_compression_kwargs()

f = h5py.File(tempfile.mktemp('-test.h5'),'w')
dset = f.create_dataset("myData", (100, 100), **kwargs)
print("✅")

EOF


echo -n 'Checking modtran is available...'
if ! command -v mod6c_cons &> /dev/null; then
  echo "❌: modtran 'mod6c_cons' not found in PATH"
  exit 1
else
  echo '✅'
fi

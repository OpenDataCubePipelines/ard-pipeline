#!/usr/bin/env bash

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

# The previous import of wagl should have initialised the filters.
print("Attempting hdf5 blosc compression...", end='', flush=True)
import h5py
import tempfile
f = h5py.File(tempfile.mktemp('-test.h5'),'w')
dset = f.create_dataset("myData", (100, 100), compression=32001)
print("✅")

EOF


echo -n 'Checking modtran is available...'
if ! command -v mod6c_cons &> /dev/null; then
  echo "❌: modtran 'mod6c_cons' not found in PATH"
  exit 1
else
  echo '✅'
fi

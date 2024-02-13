
@default: run

# editable and no-build-islation together will allow meson to auto-compile
# the native dependencies when running from your source directory.
#
# (Not-doing-this is probably why you're getting import errors running tests!)
#
@install-dev:
    python3 -m pip install --no-build-isolation --editable .

@build:
    docker build --platform linux/amd64 -t ard:dev .

@build-plain:
    docker build --progress=plain --platform linux/amd64 -t ard:dev .

@build-builder:
    docker build --platform linux/amd64 --target builder -t ard:builder .

# Don't specify platform. (Eg. On ARM Macs this will attempt ARM64)
@build-native:
    docker build -t ard:native .

@run:
    docker run --platform linux/amd64 -it --rm --volume "${PWD}:/tests" -w /tests ard:dev /bin/bash -l

# Run tests in Docker (Similar to Github Actions config)
@test:
    docker run -it --rm --volume "${PWD}/tests":/tests -w /tests ard:dev pytest

@root:
    docker run --platform linux/amd64 -it --rm --volume "${PWD}:/tests" --user root -w /tests ard:dev /bin/bash -l

# Run the builder stage. Good for debugging build
@run-builder: build-builder
    docker run --platform linux/amd64 -it --rm ard:builder /bin/bash -l

# Run python console in image
@py:
    docker run -it --rm --volume "${PWD}:/tests" -w /tests ard:dev python

# Export a snapshot of the inner conda environment
@take-env:
    docker run --rm --volume "${PWD}:/tests" --user root -w /tests ard:dev 'pip uninstall ard-pipeline rios python-fmask -y; conda env export > /tests/deployment/environment.yaml'

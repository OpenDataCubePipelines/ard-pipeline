# Running the ARD Pipeline on the NCI
The following provides an overview on how to run the ard-pipeline for a list of scenes on the NCI.

# Requirements
## Get Access to NCI Project Folders

Access to the following projects is required. Access can be requested on through the my nci portal - my.nci.org.au
- u46
- v10
- fj7

## Setup the ard-pipeline project

On the NCI the use of modules is preferred. Setup the project modules using the documentation given here - ./deployment/nci/README.md. The ./create-s2-module.sh script can be modified to create a module in the location specified by the user. For example, by changing the `module dir` line to `echo "module_dir = ${module_dir:=/g/data/yp75/ab7271/modules}"` I created the module a module in my specified directory.

**Load the module to setup the environment:**

`module load /g/data/yp75/ab7271/modules/modulefiles/ard-pipeline/20250129-1249`.

**As documented on the homepage README.md., install ard-pipeline the package using pip:**

```Bash
python3 -m pip install --no-build-isolation --editable .
```

**Check the code by running the ./check-code.sh file**

The environment should now be setup.

# Running the pipeline

The main command to run the pipeline is `ard_pbs --help`

```
$ ard_pbs --help

Usage: ard_pbs [OPTIONS]

  Equally partition a list of scenes across n nodes and submit n jobs into the
  PBS queue for ARD processing.

Options:
  --level1-list PATH         The input level1 scene list.
  --workdir DIRECTORY        The base output working directory.
  --logdir DIRECTORY         The base logging and scripts output directory.
  --pkgdir DIRECTORY         The base output packaged directory.
  --yamls-dir DIRECTORY      The base directory for level-1 dataset documents.
  --env PATH                 Environment script to source.
  --workers INTEGER RANGE    The number of workers to request per node.
                             [1<=x<=48]
  --nodes INTEGER            The number of nodes to request.
  --memory INTEGER           The memory in GB to request per node.
  --jobfs INTEGER            The jobfs memory in GB to request per node.
  --project TEXT             Project code to run under.  [required]
  --queue TEXT               Queue to submit the job into, eg normal, express.
  --walltime TEXT            Job walltime in `hh:mm:ss` format.
  --email TEXT               Notification email address.
  --index-datacube-env PATH  Datacube specific environment script to source.
  --archive-list PATH        UUID's of the scenes to archive.  This uses the
                             environment specified in index-datacube-env.
  --cleanup                  Clean-up work directory afterwards.
  --test                     Test job execution (Don't submit the job to the
                             PBS queue).
  --help                     Show this message and exit.
```

Create directories for logs, working files and the final products (pkgdir). E.g:

- `mkdir path/to/test/folder/workdir`
- `mkdir path/to/test/folder/logdir`
- `mkdir path/to/test/folder/pkgdir`

Three additional files are needed to run the pipeline (Note, currently one of these files must be placed in the v10 directory in order to mount this directory to the pbs job):

1) A .txt file containing paths to level one scenes
2) Path a directory containing odc-yaml files
3) An envrionment file

**NOTE - test files can be found in test-files directory**

odc-yaml files can be created from the list of scenes using `eo3-prepare` (https://github.com/opendatacube/eo-datasets/blob/develop/README.md):

```
eo3-prepare sentinel-l1 -f /g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2024/2024-12/70S145E-75S150E/S2B_MSIL1C_20241207T230339_N0511_R015_T54DXF_20241208T012152.zip --output-base .

```

Run a Job simillar to the following:

```
ard_pbs  --level1-list /g/data/yp75/ab7271/ard-pipeline-tests/test1/l1-neighbourhood.txt  \
--workdir /g/data/yp75/ab7271/ard-pipeline-tests/test1/workdir \
--logdir /g/data/yp75/ab7271/ard-pipeline-tests/test1/logdir  \
--pkgdir /g/data/yp75/ab7271/ard-pipeline-tests/test1/pkgdir \
--yamls-dir /g/data/yp75/ab7271/ard-pipeline-tests/test1/md-docs \
--env /g/data/v10/josh-ops-demo/antarctica-stuff/alex-test-ard-pipeline.env \
--workers 18 \
--nodes 1 \
--project u46 \
--walltime 01:00:00 \
--email alex.bradley@ga.gov.au
```

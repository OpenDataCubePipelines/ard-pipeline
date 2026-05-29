#!/usr/bin/env bash
set -e

# This job script is recommended for running single scene tests with
# a pre-built conda dev environment with `ARD pipeline` & dependencies.

scene_file=""
mission="LS"  # or "S2" for Sentinel-2, used to determine which ancillary fetch function to use
home_dir=$HOME  # used for luigi config paths, e.g. gverify executable, should be the same as the home dir in the conda environment
local_ancillary_files=""  # local dir to sync data to, e.g. for ancillary files
era5_dir_path=""  # path to era5 ancillary data, used for luigi config
merra2_dir_path=""  # path to merra2 ancillary data, used for luigi config
brdf_dir_path=""  # path to brdf ancillary data, used for luigi config
skip_gqa="true"  # whether to skip gqa, set to "skip_gqa = true" in luigi config if true
downloader_conda_path="$HOME/miniconda3/envs/ard-dl"  # name of conda env for the ancillary downloader
downloader_scripts_dir="$HOME/ard-pipeline-downloader"  # path to ancillary downloader scripts
pipeline_conda_path="$HOME/miniconda3/envs/ard-pipeline"  # name of conda env for the pipeline, used for loading the env into the "session" before running luigi
conda_base_path="$HOME/miniconda3"  # base path to conda, used for sourcing conda.sh for activating envs

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scene-file) scene_file="$2"; shift ;;
        --mission) mission="$2"; shift ;;
        --local-ancillary-files) local_ancillary_files="$2"; shift ;;
        --home-dir) home_dir="$2"; shift ;;
        --era5-dir-path) era5_dir_path="$2"; shift ;;
        --merra2-dir-path) merra2_dir_path="$2"; shift ;;
        --brdf-dir-path) brdf_dir_path="$2"; shift ;;
        --downloader-conda-path) downloader_conda_path="$2"; shift ;;
        --downloader-scripts-dir) downloader_scripts_dir="$2"; shift ;;
        --pipeline-conda-path) pipeline_conda_path="$2"; shift ;;
        --conda-base-path) conda_base_path="$2"; shift ;;
        --skip-gqa) skip_gqa="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if 'scene_file' is provided
if [[ -z "$scene_file" ]]; then
    echo "Error: --scene-file is a required parameter."
    exit 1
fi

# Assert scene file exists
if [[ ! -f "$scene_file" ]]; then
    echo "Error: Scene file '$scene_file' does not exist."
    exit 1
fi

# Assert the downloader dir exists
if [[ ! -d "$downloader_scripts_dir" ]]; then
    echo "Error: Downloader scripts dir '$downloader_scripts_dir' does not exist."
    exit 1
fi

filename=$(basename "$scene_file")
echo "Processing scene file: $scene_file with filename: $filename"

file_base=${filename%.*}

if [[ -z "$local_ancillary_files" ]]; then
    local_ancillary_files="$home_dir/ard-pipeline/test_aws/data"  # local dir to sync data to, e.g. for ancillary files
fi

if [[ -z "$era5_dir_path" ]]; then
    era5_dir_path="$local_ancillary_files/ERA5"  # path to era5 ancillary data, used for luigi config
fi

if [[ -z "$merra2_dir_path" ]]; then
    merra2_dir_path="$local_ancillary_files/MERRA2"  # path to merra2 ancillary data, used for luigi config
fi

if [[ -z "$brdf_dir_path" ]]; then
    brdf_dir_path="$local_ancillary_files/BRDF"  # path to brdf ancillary data, used for luigi config
fi

LUIGI_CONFIG_TEMPLATE="$home_dir/ard-pipeline/deployment/templates/luigi.cfg.template"  # template luigi config file, should be in the same dir as this script
LUIGI_LOGGING_TEMPLATE="$home_dir/ard-pipeline/deployment/templates/luigi-logging.cfg.template"  # template luigi logging config file, should be in the same dir as this script
S3_ANCILLARY_FILES="s3://dea-non-public-data/eo-ancillary-data"  # base path to ancillary files on s3, used for fetching ancillaries
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

inputs_file="$SCRIPT_DIR/inputs.txt"
echo "Scene file: $scene_file"
echo "Inputs file: $inputs_file"
echo "writing scene file to inputs file for luigi task..."
echo "$scene_file" > $inputs_file

echo "Using local ancillary files dir: $local_ancillary_files"
echo "Using s3 ancillary files dir: $S3_ANCILLARY_FILES"
echo "Using mission: $mission"
echo "Using ERA5 dir path: $era5_dir_path"
echo "Using MERRA2 dir path: $merra2_dir_path"
echo "Using BRDF dir path: $brdf_dir_path"
echo "Script dir: $SCRIPT_DIR"

## Generate luigi logging config file from template, replacing placeholders with actual paths
luigi_logging_config_lines=$(cat $LUIGI_LOGGING_TEMPLATE)
luigi_logging_config_lines=$(echo "$luigi_logging_config_lines" | sed "s|{{INTERFACE_FILE}}|'$SCRIPT_DIR/luigi-interface.log'|g")
luigi_logging_config_lines=$(echo "$luigi_logging_config_lines" | sed "s|{{TASK_LOG_FILE}}|'$SCRIPT_DIR/task-log.jsonl'|g")
luigi_logging_config_lines=$(echo "$luigi_logging_config_lines" | sed "s|{{WAGL_LOG_FILE}}|'$SCRIPT_DIR/status-log.jsonl'|g")

luigi_logging_file="$SCRIPT_DIR/luigi-logging.cfg"  # must be the absolute path
echo "$luigi_logging_config_lines" > $luigi_logging_file
echo "Generated luigi logging config file at $luigi_logging_file"

luigi_config_file_lines=$(cat $LUIGI_CONFIG_TEMPLATE)
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{LOGGING_CONF_FILE}}|$luigi_logging_file|g")
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{HOME_DIR}}|$home_dir|g")
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{TASK_HISTORY_DB_CONNECTION}}|$SCRIPT_DIR/luigi-task-hist.db|g")
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{ERA5_DIR_PATH}}|$era5_dir_path|g")
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{MERRA2_DIR_PATH}}|$merra2_dir_path|g")
luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{BRDF_DIR_PATH}}|$brdf_dir_path|g")

function is_empty_dir() {
  [ -z "$(ls -A "$1")" ]
}

function fetch_wrs {
  # fetch wrs tifs, if non found for this tile, fall back to backups
  path=$1
  row=$2
  dest_dir="$local_ancillary_files/GQA/wrs2/$path/$row"

  aws s3 sync "$S3_ANCILLARY_FILES/GCP/GQA_v3/wrs2/$path/$row" $dest_dir
  luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{GQA_GCP_DIR}}|$local_ancillary_files/GQA/wrs2|g")

  if is_empty_dir $dest_dir; then
    # get fall backs
    aws s3 sync "$S3_ANCILLARY_FILES/GCP/GLS2000_GCP_SCENE/wrs2/$path/$row" "$local_ancillary_files/GQA/GLS2000_GCP_SCENE/wrs2/$path/$row"
    luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{GQA_BACKUP_GCP_DIR}}|$local_ancillary_files/GQA/GLS2000_GCP_SCENE/wrs2|g")
  fi

  if is_empty_dir "$local_ancillary_files/GQA/GLS2000_GCP_SCENE/wrs2/$path/$row"; then
    echo "Warning: No WRS files found for path '$path' row '$row' in both primary and backup locations."
    luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{SKIP_GQA}}|skip_gqa = true|g")
 else
    luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{SKIP_GQA}}|# skip_gqa = true|g")
  fi
}

function fetch_era5 {
    python $downloader_scripts_dir/era5_cli.py --inputs $file_base --base-dir $era5_dir_path
}

function fetch_merra2 {
    # Placeholder function for fetching MERRA-2 ancillary data, to be implemented as needed.
}

function fetch_brdf {
    python $downloader_scripts_dir/brdf_cli.py --inputs $file_base --base-dir $brdf_dir_path
}


function fetch_ancillaries {
    conda activate $downloader_conda_path

    fetch_era5
    # fetch_merra2 $scene_date
    fetch_brdf

    conda deactivate

}


function get_path_row {
    mapfile -t OVERLAPS < <(python /scripts/tile_converter.py $1) # where is this script??
}


function fetch_ancillaries_s2 {
    echo "fetching ancillaries..."

    # get relevant segments (in correct formats)
    # e.g. S2A_MSIL1C_20260109T020511_N0511_R017_T50KQV_20260109T055857
    IFS='_' read -r _ _ date _ _ tile _ <<< "$filename"

    tile="${tile:1}"

    year="${date:0:4}"
    scene_date="${year}-${date:4:2}-${date:6:2}"

    echo "[Scene details] DATE:'$scene_date' TILE:'$tile'"

    fetch_ancillaries

    if [[ "$skip_gqa" == "true" ]]; then
        echo "Skipping GQA ancillary fetch due to missing WRS files for this scene."
        luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{SKIP_GQA}}|skip_gqa = true|g")
    else
        get_path_row $tile
        for overlap in "${OVERLAPS[@]}"
            do
                path="${overlap:0:3}"
                row="${overlap:3:3}"
                echo "[Scene details] PATH:'$path' ROW:'$row'"
                # fetch wrs ancillaries for each tile
                fetch_wrs $path $row
                # fetch Fix_QA_points file for root_fix_qa_location for each tile
                aws s3 sync "$S3_ANCILLARY_FILES/GCP/Fix_QA_points/$path/$row" "$local_ancillary_files/GQA/Fix_QA_points/$path/$row"
            done
    fi
    luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{GQA_FIX_QA_DIR}}|$local_ancillary_files/GQA/Fix_QA_points|g")

    echo "ancillary fetch completed"
}

function fetch_ancillaries_ls {
    echo "fetching ancillaries..."

    # get relevant segments (in correct formats)
    IFS='_' read -r _ _ pathrow date _ <<< "$filename"

    path="${pathrow:0:3}"
    row="${pathrow:3:3}"
    year="${date:0:4}"
    scene_date="${year}-${date:4:2}-${date:6:2}"

    echo "[Scene details] DATE:'$scene_date' PATH:'$path' ROW:'$row'"

    fetch_ancillaries

    if [[ "$skip_gqa" == "true" ]]; then
        echo "Skipping GQA ancillary fetch due to missing WRS files for this scene."
        luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{SKIP_GQA}}|skip_gqa = true|g")
    else
        fetch_wrs $path $row
        # fetch Fix_QA_points file for root_fix_qa_location
        aws s3 sync "$S3_ANCILLARY_FILES/GCP/Fix_QA_points/$path/$row" "$local_ancillary_files/GQA/Fix_QA_points/$path/$row"
    fi
    luigi_config_file_lines=$(echo "$luigi_config_file_lines" | sed "s|{{GQA_FIX_QA_DIR}}|$local_ancillary_files/GQA/Fix_QA_points|g")

    echo "ancillary fetch completed"
}

source "$conda_base_path/etc/profile.d/conda.sh"

if [ "$mission" == "LS" ]; then
    fetch_ancillaries_ls
elif [ "$mission" == "S2" ]; then
    fetch_ancillaries_s2
else
    echo "unsupported mission '$mission' for ancillary fetching"
    exit 1
fi

luigi_config_path="$SCRIPT_DIR/luigi.cfg"  # must be the absolute path
echo "$luigi_config_file_lines" > $luigi_config_path
echo "Generated luigi config file at $luigi_config_path"
export LUIGI_CONFIG_PATH=$luigi_config_path

# load the conda environment into the "session"
conda activate $pipeline_conda_path

umask 0022

# Configure output dirs in the absence of using `ard-pbs`
# note this works on date based output dirs instead of the
# ard_pbs practice of saving to files with random IDs. This
# is based on the assumption that production runs will use
# `ard_pbs` & potentially non-unique IDs here are only for
# development & debugging purposes.
project=`realpath $(pwd)`
timestamp=`date +"%Y%m%d-%H%M%S"`
work_batch_dir="$SCRIPT_DIR/work/$timestamp-batch"
log_job_dir="$SCRIPT_DIR/logs/$timestamp-batch"

luigid --background --logdir $log_job_dir
# luigid --logdir $log_job_dir

luigi --module tesp.workflow ARDP \
      --level1-list "$SCRIPT_DIR/inputs.txt" \
      --workdir $work_batch_dir \
      --pkgdir "$SCRIPT_DIR/pkg" \
      --yamls-dir="" \
      --workers 2 \
      --parallel-scheduling

# TODO optional, add results postprocessing such as:
# /some/dir/repos/ard-pipeline-ext/scripts/reflectance_post.sh $work_batch_dir

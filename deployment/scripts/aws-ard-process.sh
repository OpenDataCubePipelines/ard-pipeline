#!/bin/bash

# $0: this script
# $1: $SOURCE_S3_URL: URL of the source of the L2 data
# $2: $DESTINATION_S3_URL: URL of the destination of the L2 data

SOURCE_S3_URL="$1"
DESTINATION_S3_URL="$2"
TASK_UUID="$3"

ESTIMATED_COMPLETION_TIME=10800   # 3 hours
WORKDIR="/granules"
OUTDIR="/output"
PKGDIR="/upload"
MOD6=/ancillary/MODTRAN6.0.2.3G/bin/linux/mod6c_cons
# WORKDIR="./granules" # TODO BRAD
# OUTDIR="./output" # TODO BRAD
# PKGDIR="./upload" # TODO BRAD
# MOD6=./temp_code/mod6c_cons # TODO BRAD
export LUIGI_CONFIG_PATH="/configs/landsat.cfg" # TODO BRAD

# separate s3://bucket_name/prefix into bucket_name prefix
read SOURCE_BUCKET SOURCE_PREFIX <<< $(echo "$SOURCE_S3_URL" | perl -pe's/s3:\/\/([^\/]+)\/(.*)/\1 \2/;')
read DESTINATION_BUCKET DESTINATION_PREFIX <<< $(echo "$DESTINATION_S3_URL" | perl -pe's/s3:\/\/([^\/]+)\/(.*)/\1 \2/;')

# TODO BRAD source /scripts/lib.sh
source lib.sh
LOG_LEVEL=$LOG_DEBUG

if [ -z "$SOURCE_BUCKET" ] || [ -z "$SOURCE_PREFIX" ]; then
    log_message $LOG_INFO "[s3 source config] invalid or missing values BUCKET:'$SOURCE_BUCKET' PREFIX:'$SOURCE_PREFIX'"
    exit -1;
fi

if [ -z "$DESTINATION_BUCKET" ] || [ -z "$DESTINATION_PREFIX" ]; then
    log_message $LOG_INFO "[s3 destination config] invalid or missing values BUCKET:'$SOURCE_BUCKET' PREFIX:'$SOURCE_PREFIX'"
    exit -1;
fi

log_message $LOG_INFO "[s3 source config] BUCKET:'$SOURCE_BUCKET' PREFIX:'$SOURCE_PREFIX'"
log_message $LOG_INFO "[s3 destination config] BUCKET:'$DESTINATION_BUCKET' PREFIX:'$DESTINATION_PREFIX'"

create_task_folders
fetch_ard_granule

# Create work file
echo "$WORKDIR/$TASK_UUID" > "$WORKDIR/$TASK_UUID/scenes.txt"

check_output_exists

cd /scripts

activate_modtran ???
run_luigi

# write_stac_metadata
# upload_landsat

#remove_workdirs

finish_up

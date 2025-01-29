#!/bin/bash

# $0: this script
# $1: $SQS_QUEUE: URL of the SQS queue to fetch task from
# $2: $SOURCE_BUCKET: URL of the source bucket for L1 data
# $3: $DESTINATION_S3_URL: URL of the destination of the L2 data
# $4: $SNS_TOPIC: URL to publish ODC metadata of the L2 dataset
# $5: $EXPLORER_URL: URL for destination STAC metadata

SQS_QUEUE="$1"
SOURCE_BUCKET="$2"
DESTINATION_S3_URL="$3"
SNS_TOPIC="$4"
EXPLORER_URL="$5"

ESTIMATED_COMPLETION_TIME=10800   # 3 hours
WORKDIR="/granules"
OUTDIR="/output"
PKGDIR="/upload"
MOD6=/ancillary/MODTRAN6.0.2.3G/bin/linux/mod6c_cons
export LUIGI_CONFIG_PATH="/configs/sentinel-2.cfg"

# separate s3://bucket_name/prefix into bucket_name prefix
read DESTINATION_BUCKET DESTINATION_PREFIX <<< $(echo "$DESTINATION_S3_URL" | perl -pe's/s3:\/\/([^\/]+)\/(.*)/\1 \2/;')

source /scripts/lib.sh
LOG_LEVEL=$LOG_DEBUG

log_message $LOG_INFO "$0 called with $SQS_QUEUE $SOURCE_BUCKET $DESTINATION_S3_URL $SNS_TOPIC $EXPLORER_URL"
log_message $LOG_INFO "[s3 destination config] BUCKET:'$DESTINATION_BUCKET' PREFIX:'$DESTINATION_PREFIX'"

# saves the message to $WORKDIR/message.json and the body to $WORKDIR/task.json
receive_message

RECEIPT_HANDLE=$(jq -r '.Messages[0].ReceiptHandle' "$WORKDIR/message.json")
GRANULE_PATH=$(jq -r '.tiles[0].path' "$WORKDIR/task.json")
DATASTRIP_PATH=$(jq -r '.tiles[0].datastrip.path' "$WORKDIR/task.json")

TASK_UUID=$(jq -r '.id' "$WORKDIR/task.json")
GRANULE_URL="s3://${SOURCE_BUCKET}/${GRANULE_PATH}"
DATASTRIP_URL="s3://${SOURCE_BUCKET}/${DATASTRIP_PATH}"

log_message $LOG_INFO "TASK_UUID=${TASK_UUID}"
log_message $LOG_INFO "GRANULE_URL=${GRANULE_URL}"
log_message $LOG_INFO "DATASTRIP_URL=${DATASTRIP_URL}"

create_task_folders
fetch_sentinel2_granule

# Create work file
echo "$WORKDIR/$TASK_UUID" > "$WORKDIR/$TASK_UUID/scenes.txt"

prepare_level1_sentinel_2
check_output_exists

cd /scripts

activate_modtran
run_luigi

write_stac_metadata
upload_sentinel2
delete_message
publish_sns
remove_workdirs

finish_up

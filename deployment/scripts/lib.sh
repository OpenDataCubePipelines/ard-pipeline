LOG_DEBUG=1
LOG_INFO=10
LOG_WARN=100
LOG_ERROR=1000

# Logging interface similar to python logging
function log_message {
    if [ "$1" -ge $LOG_LEVEL ]; then
        echo "$2"
    fi
}

# Log contents of a stream
function log_stream {
    while IFS='' read -r line || [[ -n "$line" ]];
    do
        log_message "$1" "$line"
    done
}

# Create task-specific folders
function create_task_folders {
    mkdir -p "$WORKDIR/$TASK_UUID"
    mkdir -p "$OUTDIR/$TASK_UUID"
    mkdir -p "$PKGDIR/$TASK_UUID"
}

function remove_workdirs {
    # Remove referenced data ahead of time since the docker orchestrator may be
    # delayed in freeing up storage for re-use
    rm -rf "$PKGDIR/$TASK_UUID"
    rm -rf "$OUTDIR/$TASK_UUID"
    rm -rf "$WORKDIR/$TASK_UUID"
    log_message $LOG_INFO "Working directories removed"
}

# Fetch Landsat granule from S3 bucket
function fetch_landsat_granule {
    log_message $LOG_INFO "Fetching scene from s3"
    aws s3 sync --only-show-errors "$L1_BUCKET/$L1_PREFIX" "$WORKDIR/$TASK_UUID"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Unable to fetch scene";
        exit -1;
    fi
    find "$WORKDIR/$TASK_UUID" -type f
    log_message $LOG_INFO "Fetching scene completed"
}

# Fetch Sentinel-2 granule from S3 bucket
function fetch_sentinel2_granule {
    log_message $LOG_INFO "Syncing tile information from s3"
    # Sync data for the current granule
    aws s3 sync --only-show-errors "$GRANULE_URL" "$WORKDIR/$TASK_UUID"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Unable to fetch granule";
        exit -1;
    fi
    log_message $LOG_INFO "Tile synched"

    aws s3 sync --only-show-errors "$DATASTRIP_URL" "$WORKDIR/$TASK_UUID/datastrip"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Unable to fetch metadata";
        exit -1;
    fi
    log_message $LOG_INFO "Metadata synched"

    # top level metadata if there
    aws s3 cp --only-show-errors "${DATASTRIP_URL%/datastrip/0}/metadata.xml" "$WORKDIR/$TASK_UUID/l1c-metadata.xml"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Unable to fetch top-level metadata, skipping";
    else
        # If data includes radiation offsets, remove them and their record in the MTD xml
        log_message $LOG_INFO "Rewriting Data with radiation offsets (if required)"
        /scripts/remove_l1_dn_offsets.sh "$WORKDIR/$TASK_UUID"
    fi

    find "$WORKDIR/$TASK_UUID" -type f
    log_message $LOG_INFO "Fetching scene completed"
}

# Run luigi task ARDP
function run_luigi {
    log_message $LOG_INFO "Running luigi"
    luigi --module tesp.workflow ARDP \
        --level1-list "$WORKDIR/$TASK_UUID/scenes.txt" \
        --workdir "$OUTDIR/$TASK_UUID" \
        --pkgdir "$PKGDIR/$TASK_UUID" \
        --local-scheduler \
        --workers 1

    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Luigi failed: status-log.jsonl"
        log_stream $LOG_ERROR < "status-log.jsonl"

        log_message $LOG_ERROR "Luigi failed: task-log.jsonl"
        log_stream $LOG_ERROR < "task-log.jsonl"

        log_message $LOG_ERROR "Luigi failed: luigi-interface.log"
        log_stream $LOG_ERROR < "luigi-interface.log"

        # Cleanup
        log_message $LOG_INFO "Processing failed, removing working directories";
        remove_workdirs
        exit -1;
    fi
}

# Activate modtran6 license
function activate_modtran {
    log_message $LOG_INFO "Activating MODTRAN6"
    $MOD6 -version
    $MOD6 -activate_license $MODTRAN_PRODUCT_KEY
}

# Check if output already exists
function check_output_exists {
    python /scripts/check-exists.py --level1-path="$WORKDIR/$TASK_UUID" --s3-bucket="$DESTINATION_BUCKET" --s3-prefix="$DESTINATION_PREFIX"
    if [ "$?" -eq 0 ]; then
        log_message $LOG_INFO "Output already exists, exiting"
	if [[ -v RECEIPT_HANDLE ]]; then
            delete_message
	fi
        exit 0;
    fi
}

# Receive task from queue
function receive_message {
    aws sqs receive-message --queue-url="$SQS_QUEUE" --visibility-timeout="$ESTIMATED_COMPLETION_TIME" --max-number-of-messages=1 > "$WORKDIR/message.json" 2> "$WORKDIR/message.err"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not receive message from ${SQS_QUEUE}"
        log_stream $LOG_ERROR < "$WORKDIR/message.err"
        exit -1;
    else
        log_message $LOG_DEBUG "Received message from ${SQS_QUEUE}"
        log_stream $LOG_DEBUG < "$WORKDIR/message.json"
    fi


    if [ -s "$WORKDIR/message.json" ]; then
        # Extract the task from the message
        jq -r '.Messages[0].Body' "$WORKDIR/message.json" | jq -r '.Message' > "$WORKDIR/task.json"
    else
        log_message $LOG_INFO "No messages in the queue, nothing to do!"
        log_message $LOG_INFO "Exiting with exit code 0"
	exit 0;
    fi
}

# Prepare level-1 dataset yaml for sentinel-2
function prepare_level1_sentinel_2 {
# Note that argument refers to a filename and not a directory
    log_message $LOG_INFO "Generating 1C product metadata"
    eo3-prepare sentinel-l1 "$WORKDIR/$TASK_UUID" --output-base "$WORKDIR/$TASK_UUID"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not prepare level-1 metadata";
        exit -1;
    fi
    CAPTURE_DATE="$(date -u --date=$(cat "$WORKDIR/$TASK_UUID/productInfo.json" | jq -r '.tiles[0].timestamp') '+%Y-%m-%d')"
    log_message $LOG_INFO "Generated 1C product metadata"
    log_message $LOG_INFO "CAPTURE_DATE=$CAPTURE_DATE"
}

# Prepare level-1 dataset yaml for landsat
function prepare_level1_landsat {
    log_message $LOG_INFO "Generating 1C product metadata"
    eo3-prepare landsat-l1 $WORKDIR/$TASK_UUID
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not prepare level-1 metadata";
        exit -1;
    fi
    mv $WORKDIR/*.yaml $WORKDIR/$TASK_UUID
    log_message $LOG_INFO "Generated 1C product metadata"
}

function write_stac_metadata {
    # create STAC item for this dataset
    local PARENT_DIR=$(dirname $(find "$PKGDIR/$TASK_UUID" -type f -name '*.odc-metadata.yaml' -printf '%P\n'))
    log_message $LOG_INFO "Parent dir for documents ${PARENT_DIR}"
    pushd "$PKGDIR/$TASK_UUID/${PARENT_DIR}"
    eo3-to-stac -v --validate -u ${DESTINATION_S3_URL}${PARENT_DIR}/ -e ${EXPLORER_URL} *.odc-metadata.yaml
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not create STAC item";
        exit -1;
    fi
    rm -f stac_schema_cache.sqlite
    log_message $LOG_INFO "Created STAC item $(ls *.json)"
    popd
}

function upload_sentinel2 {
    # upload to destination
    log_message $LOG_INFO "Copying to destination"
    aws s3 cp --recursive --only-show-errors --acl bucket-owner-full-control "$PKGDIR/$TASK_UUID" "${DESTINATION_S3_URL}"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not upload s3 objects";
        exit -1;
    fi
    find "$PKGDIR/$TASK_UUID" -type f -printf '%P\n' | xargs -n 1 -I {} aws s3api put-object-tagging --bucket "${DESTINATION_BUCKET}" --tagging 'TagSet=[{Key=pipeline,Value="NRT Processing"},{Key=target_data,Value="Sentinel2 NRT"},{Key=remote_host,Value="AWS PDS Europe"},{Key=transfer_method,Value="Public Internet Fetch"},{Key=input_data,Value="Sentinel2 L1C"},{Key=input_data_type,Value="JP2000"},{Key=egress_location,Value="ap-southeast-2"},{Key=egress_method,Value="s3 upload"},{Key=archive_time,Value="30 days"},{Key=orchestrator,Value="KEDA"}]' --key "${DESTINATION_PREFIX}"{}
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not update s3 object tags";
        exit -1;
    fi
    log_message $LOG_INFO "Synch to destination complete"
}

function upload_landsat {
    # upload to destination
    log_message $LOG_INFO "Copying to destination"
    aws s3 cp --recursive --only-show-errors --acl bucket-owner-full-control "$PKGDIR/$TASK_UUID" "${DESTINATION_S3_URL}"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not upload s3 objects";
        exit -1;
    fi
    find "$PKGDIR/$TASK_UUID" -type f -printf '%P\n' | xargs -n 1 -I {} aws s3api put-object-tagging --bucket "${DESTINATION_BUCKET}" --tagging 'TagSet=[{Key=pipeline,Value="NRT Processing"},{Key=target_data,Value="Landsat NRT"},{Key=remote_host,Value="USGS M2M API"},{Key=transfer_method,Value="Internet Transfer"},{Key=input_data,Value="Landsat L1RT"},{Key=input_data_type,Value="GeoTIFF"},{Key=egress_location,Value="ap-southeast-2"},{Key=egress_method,Value="s3 upload"},{Key=archive_time,Value="30 days"},{Key=orchestrator,Value="KEDA"}]' --key "${DESTINATION_PREFIX}"{}
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not update s3 object tags";
        exit -1;
    fi
    log_message $LOG_INFO "Synch to destination complete"
}

# Acknowledge success by deleting the SQS message
function delete_message {
    log_message $LOG_INFO "Deleting message"
    aws sqs delete-message --queue-url="$SQS_QUEUE" --receipt-handle="$RECEIPT_HANDLE"
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not delete task after completion";
        exit -1;
    fi
    log_message $LOG_INFO "Message deleted"
}

# Publish success in an sns
function publish_sns() {
	python /scripts/publish-sns.py --stac-file="$PKGDIR/$TASK_UUID"/$(find "$PKGDIR/$TASK_UUID" -type f -name '*.stac-item.json' -printf '%P\n') --sns-arn=$SNS_TOPIC
    if [ "$?" -ne 0 ]; then
        log_message $LOG_ERROR "Could not send success SNS notification";
        exit -1;
    fi
    log_message $LOG_INFO "SNS notification published"
}

# Finish up script run
function finish_up {
    log_message $LOG_INFO "Complete"
    log_message $LOG_INFO "Exiting with exit code 0"
    exit 0;
}

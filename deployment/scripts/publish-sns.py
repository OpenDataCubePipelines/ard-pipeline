#!/usr/bin/env python3

import json
import math

import boto3
import click


def get_attributes(stac_doc):
    properties = stac_doc["properties"]
    bbox = stac_doc["bbox"]
    gqa_iterative_mean_xy = properties["gqa:iterative_mean_xy"]

    return {
        "action": {"DataType": "String", "StringValue": "ADDED"},
        "datetime": {
            "DataType": "String",
            "StringValue": properties["datetime"],
        },
        "product": {
            "DataType": "String",
            "StringValue": properties.get("odc:product", stac_doc["collection"]),
        },
        "region_code": {
            "DataType": "String",
            "StringValue": properties["odc:region_code"],
        },
        "bbox.ll_lon": {"DataType": "Number", "StringValue": str(bbox[0])},
        "bbox.ll_lat": {"DataType": "Number", "StringValue": str(bbox[1])},
        "bbox.ur_lon": {"DataType": "Number", "StringValue": str(bbox[2])},
        "bbox.ur_lat": {"DataType": "Number", "StringValue": str(bbox[3])},
        **(
            {
                "gqa_iterative_mean_xy": {
                    "DataType": "Number",
                    "StringValue": str(gqa_iterative_mean_xy),
                }
            }
            if gqa_iterative_mean_xy is not None
            and not math.isnan(gqa_iterative_mean_xy)
            else {}
        ),
        "maturity": {
            "DataType": "String",
            "StringValue": properties["dea:dataset_maturity"],
        },
        "cloudcover": {
            "DataType": "Number",
            "StringValue": str(properties["eo:cloud_cover"]),
        },
    }


def publish_sns(sns_arn, stac_doc, msg_attrs):
    client = boto3.client("sns")
    client.publish(
        TopicArn=sns_arn,
        Message=json.dumps(stac_doc, indent=4),
        MessageAttributes=msg_attrs,
    )


@click.command()
@click.option("--stac-file", required=True)
@click.option("--sns-arn", required=True)
def main(stac_file, sns_arn):
    with open(stac_file) as fl:
        stac_doc = json.load(fl)

    msg_attrs = get_attributes(stac_doc)
    publish_sns(sns_arn, stac_doc, msg_attrs)


if __name__ == "__main__":
    main()

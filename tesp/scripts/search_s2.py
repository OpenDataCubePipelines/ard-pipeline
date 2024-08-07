#!/usr/bin/env python3
"""Query available S2 L1C data for an area of interest of target input zip archives.

example usage:

    python search_s2.py --bounds 111 156 -45 -8 --date_start 01/11/2015 --date_stop 01/12/2015
    --product s2a_level1c_granule --config ~/.aws_datacube.conf --output /g/data/v10/tmp/output.txt
"""

import logging
import os
from datetime import datetime

import click
import datacube
from click_datetime import Datetime
from datacube import Datacube
from datacube.model import Range


@click.command(help=__doc__)
@click.option(
    "--output",
    help="Output directory",
    prompt=True,
    type=click.Path(exists=False, writable=True, dir_okay=True),
)
@click.option(
    "--bounds",
    help="Bounding coordinates in longitude and latitude <lon_min> <lon_max> <lat_min> <lat_max>",
    type=float,
    nargs=4,
)
@click.option(
    "--date_start",
    prompt=True,
    type=Datetime(format="%d/%m/%Y"),
    default=datetime.now(),
    help="Search start date DD/MM/YYYY",
)
@click.option(
    "--date_stop",
    prompt=True,
    type=Datetime(format="%d/%m/%Y"),
    default=datetime.now(),
    help="Search end date DD/MM/YYYY",
)
@click.option("--product", help="Datacube Product / Data Type", prompt=True, nargs=1)
@click.option(
    "--config",
    help="Datacube configuration file if using an external or test datacube",
    type=click.Path(exists=True, readable=True, writable=False),
    nargs=1,
)
def main(output, bounds, date_start, date_stop, config, product):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    if config is not None:
        dc = Datacube(config=config)
    else:
        dc = Datacube()

    filename = (
        product
        + "_"
        + str(bounds[0])
        + "_"
        + str(bounds[1])
        + "_"
        + str(bounds[2])
        + "_"
        + str(bounds[3])
        + "_"
        + str(date_start).replace(" ", "")
        + "_"
        + str(date_stop).replace(" ", "")
        + ".txt"
    )
    outfile = os.path.join(output, filename)

    query = datacube.api.query.Query(
        product=product,
        time=Range(date_start, date_stop),
        longitude=(bounds[0], bounds[1]),
        latitude=(bounds[2], bounds[3]),
    )
    logging.info("Search results output to %s", outfile)
    data = dc.index.datasets.search_eager(**query.search_terms)
    # turn output into a uniq list of paths to the L1C zip data
    zip_list = []
    for dataset in data:
        zip_list.append(
            dataset.metadata_doc["image"]["bands"]["B01"]["path"]
            .split("!")[0]
            .replace("zip:", "")
        )
    unique = list(set(zip_list))

    search_result = open(outfile, "w")
    for item in unique:
        search_result.write(f"{item}\n")


if __name__ == "__main__":
    main()

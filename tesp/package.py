#!/usr/bin/env python
# pylint: disable=too-many-locals

import json
import shutil
import uuid
from pathlib import Path
from posixpath import join as ppjoin
from typing import Tuple

import eodatasets3.wagl
import h5py
import numpy as np
import rasterio
import yaml
from affine import Affine
from boltons.iterutils import get_path
from datacube.utils import jsonify_document
from eodatasets3 import DatasetAssembler, images, utils
from eodatasets3.scripts.tostac import dc_to_stac, json_fallback
from eodatasets3.serialise import from_path, loads_yaml
from eodatasets3.wagl import Granule
from rasterio.crs import CRS
from yaml.representer import Representer

from wagl.hdf5 import find

yaml.add_representer(np.int8, Representer.represent_int)
yaml.add_representer(np.uint8, Representer.represent_int)
yaml.add_representer(np.int16, Representer.represent_int)
yaml.add_representer(np.uint16, Representer.represent_int)
yaml.add_representer(np.int32, Representer.represent_int)
yaml.add_representer(np.uint32, Representer.represent_int)
yaml.add_representer(int, Representer.represent_int)
yaml.add_representer(np.int64, Representer.represent_int)
yaml.add_representer(np.uint64, Representer.represent_int)
yaml.add_representer(float, Representer.represent_float)
yaml.add_representer(np.float32, Representer.represent_float)
yaml.add_representer(np.float64, Representer.represent_float)
yaml.add_representer(np.ndarray, Representer.represent_list)


def package_non_standard(
    base_output_dir: Path, granule: Granule
) -> Tuple[uuid.UUID, Path]:
    """
    A prototype package alternative that uses wagl's native H5 for storage.

    It will copy the hdf5-file to the output as-is (using dataset subfolders), and
    generate suitable yaml files for ODC.
    """
    input_hdf5: Path = granule.wagl_hdf5
    assert input_hdf5.exists()

    with DatasetAssembler(
        collection_location=base_output_dir, naming_conventions="dea"
    ) as da:
        level1 = granule.source_level1_metadata
        da.add_source_dataset(
            level1, auto_inherit_properties=True, inherit_geometry=True
        )
        da.product_family = "ard"
        da.producer = "ga.gov.au"
        da.properties["odc:file_format"] = "HDF5"

        with h5py.File(input_hdf5, "r") as fid:
            img_paths = [ppjoin(fid.name, pth) for pth in find(fid, "IMAGE")]
            granule_group = fid[granule.name]

            try:
                wagl_path, *ancil_paths = (
                    pth for pth in find(granule_group, "SCALAR") if "METADATA" in pth
                )
            except ValueError:
                raise ValueError("No nbar metadata found in granule")

            [wagl_doc] = loads_yaml(granule_group[wagl_path][()])

            da.processed = get_path(wagl_doc, ("system_information", "time_processed"))

            platform = da.properties["eo:platform"]
            if platform == "sentinel-2a" or platform == "sentinel-2b":
                org_collection_number = 3
            else:
                org_collection_number = utils.get_collection_number(
                    platform, da.producer, da.properties["landsat:collection_number"]
                )

            da.dataset_version = f"{org_collection_number}.1.0"
            da.region_code = eodatasets3.wagl._extract_reference_code(da, granule.name)

            eodatasets3.wagl._read_gqa_doc(da, granule.gqa_doc)
            eodatasets3.wagl._read_fmask_doc(da, granule.fmask_doc)

            output_dir = da._work_path

            wagl_h5: Path = output_dir / (granule.name + ".wagl.h5")
            # Copy data from input to output
            shutil.copytree(input_hdf5.parent, output_dir, dirs_exist_ok=True)

            fmask_img = output_dir / (granule.name + ".fmask.img")
            assert fmask_img.exists()

            boolean_h5 = output_dir / (granule.name + ".converted-datasets.h5")
            created_f = h5py.File(boolean_h5, "w")

            with rasterio.open(fmask_img) as ds:
                fmask_layer = f"/{granule.name}/OA_FMASK/oa_fmask"
                data = ds.read(1)
                fmask_ds = created_f.create_dataset(
                    fmask_layer, data=data, compression="lzf", shuffle=True
                )
                fmask_ds.attrs["crs_wkt"] = ds.crs.wkt
                fmask_ds.attrs["geotransform"] = ds.transform.to_gdal()

                fmask_ds.attrs["description"] = (
                    "Converted from ERDAS Imagine format to HDF5 to work with the limitations of varied formats within ODC"  # noqa E501
                )

                grid_spec = images.GridSpec(
                    shape=ds.shape,
                    transform=ds.transform,
                    crs=CRS.from_wkt(fmask_ds.attrs["crs_wkt"]),
                )

                measurement_name = "oa_fmask"

                no_data = fmask_ds.attrs.get("no_data_value")
                if no_data is None:
                    no_data = float("nan")
                da._measurements.record_image(
                    measurement_name,
                    grid_spec,
                    boolean_h5,
                    fmask_ds[:],
                    layer=f"/{fmask_layer}",
                    nodata=no_data,
                    expand_valid_data=False,
                )

            for pathname in img_paths:
                ds = fid[pathname]
                ds_path = Path(ds.name)

                # eodatasets internally uses this grid spec to group image datasets
                grid_spec = images.GridSpec(
                    shape=ds.shape,
                    transform=Affine.from_gdal(*ds.attrs["geotransform"]),
                    crs=CRS.from_wkt(ds.attrs["crs_wkt"]),
                )

                # product group name; lambertian, nbar, nbart, oa
                if "STANDARDISED-PRODUCTS" in str(ds_path):
                    product_group = ds_path.parent.name
                elif "INTERPOLATED-ATMOSPHERIC-COEFFICIENTS" in str(ds_path):
                    product_group = f"oa_{ds_path.parent.name}"
                else:
                    product_group = "oa"

                # spatial resolution group
                # used to separate measurements with the same name
                resolution_group = "rg{}".format(ds_path.parts[2].split("-")[-1])

                measurement_name = (
                    "_".join(
                        [
                            resolution_group,
                            product_group,
                            ds.attrs.get("alias", ds_path.name),
                        ]
                    )
                    .replace("-", "_")
                    .lower()
                )  # we don't want hyphens in odc land

                # include this band in defining the valid data bounds?
                include = True if "nbart" in measurement_name else False

                no_data = ds.attrs.get("no_data_value")
                if no_data is None:
                    no_data = float("nan")

                # if we are of type bool, we'll have to convert just for GDAL
                if ds.dtype.name == "bool":
                    out_ds = created_f.create_dataset(
                        measurement_name,
                        data=np.uint8(ds[:]),
                        compression="lzf",
                        shuffle=True,
                        chunks=ds.chunks,
                    )

                    for k, v in ds.attrs.items():
                        out_ds.attrs[k] = v

                    da._measurements.record_image(
                        measurement_name,
                        grid_spec,
                        boolean_h5,
                        out_ds[:],
                        layer=f"/{out_ds.name}",
                        nodata=no_data,
                        expand_valid_data=include,
                    )
                else:
                    # work around as note_measurement doesn't allow us to specify the gridspec
                    da._measurements.record_image(
                        measurement_name,
                        grid_spec,
                        wagl_h5,
                        ds[:],
                        layer=f"/{ds.name}",
                        nodata=no_data,
                        expand_valid_data=include,
                    )

        # the longest part here is generating the valid data bounds vector
        # landsat 7 post SLC-OFF can take a really long time
        return da.done()


def write_stac_metadata(input_metadata, pkgdir, stac_base_url, explorer_base_url):
    dataset = from_path(input_metadata)
    name = input_metadata.stem.replace(".odc-metadata", "")
    output_path = input_metadata.with_name(f"{name}.stac-item.json")

    assert str(input_metadata).startswith(
        pkgdir
    ), f"was expecting {input_metadata} to start with {pkgdir}"
    stac_url = (
        stac_base_url.rstrip("/") + "/" + str(input_metadata)[len(pkgdir) :].lstrip("/")
    )

    # Create STAC dict
    item_doc = dc_to_stac(
        dataset,
        input_metadata,
        output_path,
        stac_url,
        explorer_base_url,
        do_validate=False,
    )

    with output_path.open("w") as f:
        json.dump(jsonify_document(item_doc), f, indent=4, default=json_fallback)

    return output_path

#!/usr/bin/env python

import argparse
import os
from os.path import basename, dirname, exists, splitext
from os.path import join as pjoin

import h5py
import numpy as np
import yaml
from gaip.acquisition import acquisitions
from gaip.data import write_img
from gaip.geobox import GriddedGeoBox
from gaip.hdf5 import find
from rasterio.enums import Resampling
from yaml.representer import Representer

from fmask_cophub import fmask_cogtif
from yaml_merge import merge_metadata

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

PRODUCTS = ['NBAR', 'NBART']
LEVELS = [2, 4, 8, 16, 32]


def gaip_unpack(scene, granule, h5group, outdir):
    """Unpack and package the NBAR and NBART products."""
    # listing of all datasets of IMAGE CLASS type
    img_paths = find(h5group, 'IMAGE')

    for product in PRODUCTS:
        for pathname in [p for p in img_paths if f'/{product}/' in p]:

            dataset = h5group[pathname]
            acqs = scene.get_acquisitions(group=pathname.split('/')[0],
                                          granule=granule)
            acq = [a for a in acqs if
                   a.band_name == dataset.attrs['band_name']][0]

            # base_dir = pjoin(splitext(basename(acq.pathname))[0], granule)
            base_fname = f'{splitext(basename(acq.uri))[0]}.TIF'
            out_fname = pjoin(outdir,
                              # base_dir.replace('L1C', 'ARD'),
                              # granule.replace('L1C', 'ARD'),
                              product,
                              base_fname.replace('L1C', product))

            # output
            if not exists(dirname(out_fname)):
                os.makedirs(dirname(out_fname))

            write_img(dataset, out_fname, cogtif=True, levels=LEVELS,
                      nodata=dataset.attrs['no_data_value'],
                      geobox=GriddedGeoBox.from_dataset(dataset),
                      resampling=Resampling.nearest,
                      options={'blockxsize': dataset.chunks[1],
                               'blockysize': dataset.chunks[0],
                               'compress': 'deflate',
                               'zlevel': 4})

    # retrieve metadata
    scalar_paths = find(h5group, 'SCALAR')
    pathname = [pth for pth in scalar_paths if 'NBAR-METADATA' in pth][0]
    tags = yaml.load(h5group[pathname][()])
    return tags


def main(l1_path, gaip_fname, fmask_path, yamls_path, outdir):
    """Main level."""
    scene = acquisitions(l1_path)
    with open(pjoin(yamls_path, f'{scene.label}.yaml')) as src:
        l1_documents = {doc['tile_id']: doc for doc in yaml.load_all(src)}

    with h5py.File(gaip_fname, 'r') as fid:
        for granule in scene.granules:
            if granule is None:
                h5group = fid['/']
            else:
                h5group = fid[granule]

            ard_granule = granule.replace('L1C', 'ARD')
            out_path = pjoin(outdir, ard_granule)

            # fmask cogtif conversion
            fmask_cogtif(pjoin(fmask_path, f'{granule}.cloud.img'),
                         pjoin(out_path, f'{ard_granule}_QA.TIF'))

            # unpack the data produced by gaip
            gaip_tags = gaip_unpack(scene, granule, h5group, out_path)

            # merge all the yaml documents
            tags = merge_metadata(l1_documents[granule], gaip_tags, out_path)

            with open(pjoin(out_path, 'ARD-METADATA.yaml'), 'w') as src:
                yaml.dump(tags, src, default_flow_style=False, indent=4)

            # build vrts, contiguity, quicklook, thumbnail ...


if __name__ == '__main__':
    description = "Prepare or package a gaip output."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--level1-pathname", required=True,
                        help="The level1 pathname.")
    parser.add_argument("--gaip-filename", required=True,
                        help="The filename of the gaip output.")
    parser.add_argument("--fmask-pathname", required=True,
                        help=("The pathname to the directory containing the "
                              "fmask results for the level1 dataset."))
    parser.add_argument("--prepare-yamls", required=True,
                        help="The pathname to the level1 prepare yamls.")
    parser.add_argument("--outdir", required=True,
                        help="The output directory.")

    args = parser.parse_args()

    main(args.level1_pathname, args.gaip_filename, args.fmask_pathname,
         args.prepare_yamls, args.outdir)
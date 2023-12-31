import h5py
import pytest

from wagl.constants import DatasetName
from wagl.hdf5 import VLEN_STRING
from wagl.metadata import MetadataError, current_h5_metadata, yaml


def test_h5_metadata_single(tmpdir):
    """Tests retrieving a metadata document from a simple h5 collection."""
    test_doc = {"id": "297933b9-d728-464b-be25-c405b1efa49b"}
    with h5py.File(tmpdir.join("test.h5"), "w", driver="core") as f:
        dataset_path = (
            f"/{DatasetName.METADATA.value}/{DatasetName.CURRENT_METADATA.value}"
        )
        f.create_dataset(dataset_path, (1,), dtype=VLEN_STRING)
        f[dataset_path][()] = yaml.dump(test_doc, default_flow_style=False)

        retrieve = current_h5_metadata(f)
        assert retrieve == test_doc


def test_h5_metadata_collection(tmpdir):
    """Tests retrieving a metadata document from a multidataset h5 collection."""
    h5_dataset = "/1990/JUL/0600"
    test_doc = {"id": "297933b9-d728-464b-be25-c405b1efa49v"}
    with h5py.File(tmpdir.join("test.h5"), "w", driver="core") as f:
        dataset_path = f"/{DatasetName.METADATA.value}{h5_dataset}/{DatasetName.CURRENT_METADATA.value}"
        f.create_dataset(dataset_path, (1,), dtype=VLEN_STRING)
        f[dataset_path][()] = yaml.dump(test_doc, default_flow_style=False)

        retrieve = current_h5_metadata(f, dataset_path=h5_dataset)
        assert retrieve == test_doc


def test_h5_metadata_error(tmpdir):
    """Tests exception is raised if unable to retrieve metadata document
    for h5_metadata collection.
    """
    with h5py.File(tmpdir.join("test.h5"), "w", driver="core") as f:
        with pytest.raises(MetadataError) as _:
            current_h5_metadata(f)

#!/usr/bin/env python

# flake8: noqa

"""
Contains various HDF5/h5py wrapped utilities for writing various datasets
such as images and tables, as well as attaching metadata.
"""

import datetime
import re
from functools import partial
from pprint import pprint

from posixpath import join as ppjoin, normpath
import numpy
import h5py
import pandas

from .compression import H5CompressionFilter, BloscCompression, BloscShuffle
from .compression import H5CompressionConfig, H5lzf, H5gzip, H5zstandard
from .compression import H5bitshuffle, H5mafisc, H5blosc

DEFAULT_IMAGE_CLASS = {"CLASS": "IMAGE", "IMAGE_VERSION": "1.2", "DISPLAY_ORIGIN": "UL"}

DEFAULT_TABLE_CLASS = {"CLASS": "TABLE", "VERSION": "0.2"}

DEFAULT_SCALAR_CLASS = {"CLASS": "SCALAR", "VERSION": "0.1"}


VLEN_STRING = h5py.special_dtype(vlen=str)


def _fixed_str_size(data):
    """
    Useful for a Pandas column of data that is at its base a string
    datatype. The max length of all records in this column is
    identified and returns a NumPy datatype string, which details a
    fixed length string datatype.
    Py3 NumPy fixed string defaults to wide characters (|U), which is
    unsupported by HDF5.
    """
    str_sz = data.str.len().max()
    return "|S{}".format(str_sz)


def safeguard_dtype(datatype):
    """
    Was observed under Python2 and setting unicode as the base
    datatype for string objects, where defining a custom NumPy
    named datatype resulted in TypeError's. Hence this function.
    However it isn't required when using either Python versions'
    base string datatype, i.e. Py2->bytes, Py3->unicode.
    """
    try:
        dtype = numpy.dtype(datatype)
    except TypeError:
        dtype = numpy.dtype([(bytes(name), val) for name, val in datatype])
    return dtype


def attach_image_attributes(dataset, attrs=None):
    """
    Attaches attributes to an HDF5 `IMAGE` Class dataset.

    :param dataset:
        A `NumPy` compound dataset.

    :param attrs:
        A `dict` containing any additional attributes to attach to
        the dataset.
    """
    for key in DEFAULT_IMAGE_CLASS:
        dataset.attrs[key] = DEFAULT_IMAGE_CLASS[key]

    attach_attributes(dataset, attrs)


def attach_table_attributes(dataset, title="Table", attrs=None):
    """
    Attaches attributes to an HDF5 `TABLE` Class dataset.

    :param dataset:
        A `NumPy` compound dataset.

    :param title:
        A `string` containing the title of the TABLE dataset.
        Default is `Table`.

    :param attrs:
        A `dict` containing any additional attributes to attach to
        the dataset.
    """
    for key in DEFAULT_TABLE_CLASS:
        dataset.attrs[key] = DEFAULT_TABLE_CLASS[key]

    dataset.attrs["TITLE"] = title

    # column names
    col_fmt = "FIELD_{}_NAME"
    columns = dataset.dtype.names
    for i, col in enumerate(columns):
        dataset.attrs[col_fmt.format(i)] = col

    attach_attributes(dataset, attrs)


def create_image_dataset(
    fid,
    dataset_name,
    shape,
    dtype,
    compression=H5CompressionFilter.LZF,
    attrs=None,
    filter_opts=None,
):
    """
    Initialises a HDF5 dataset populated with some basic attributes
    that detail the Image specification.
    https://support.hdfgroup.org/HDF5/doc/ADGuide/ImageSpec.html

    :param fid:
        The file id opened for writing.

    :param dataset_name:
        The name used identify the dataset. The name can contain '/'
        used to identify the full path name. Any 'GROUP' names in the
        path will be created if they don't already exist.

    :param shape:
        The shape/dimensions of the dataset to create.

    :param dtype:
        TODO.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param attrs:
        A `dict` by which the keys will be the attribute name, and the
        values will be the attribute value.

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :return:
        A h5py.Dataset with IMAGE CLASS attributes set.
    """
    kwargs = compression.settings(filter_opts)

    dset = fid.create_dataset(dataset_name, shape=shape, dtype=dtype, **kwargs)

    attach_image_attributes(dset, attrs)

    return dset


def write_h5_image(
    data,
    dset_name,
    group,
    compression=H5CompressionFilter.LZF,
    attrs=None,
    filter_opts=None,
):
    """
    Writes a `NumPy` array direct to a `h5py.Dataset` to the
    HDF5 IMAGE CLASS standard.

    :param data:
        The `NumPy` structured array containing the data to be
        written to disk.

    :param dset_name:
        A `str` containing the name and location of the dataset
        to write to.

    :param group:
        A h5py `Group` or `File` object from which to write the
        dataset to.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param attrs:
        A `dict` of key, value items to be attached as attributes
        to the `Table` dataset.

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :return:
        None
    """
    kwargs = compression.settings(filter_opts)
    dset = group.create_dataset(dset_name, data=data, **kwargs)

    # make a copy so as not to modify the users data
    attributes = {} if attrs is None else attrs.copy()

    if data.dtype.names:
        for band_name in data.dtype.names:
            attributes["{}_MINMAXRANGE".format(band_name)] = [
                numpy.nanmin(data[band_name]),
                numpy.nanmax(data[band_name]),
            ]
    else:
        attributes["IMAGE_MINMAXRANGE"] = [numpy.nanmin(data), numpy.nanmax(data)]

    attach_image_attributes(dset, attributes)


def write_h5_table(
    data,
    dset_name,
    group,
    compression=H5CompressionFilter.LZF,
    title="Table",
    attrs=None,
    filter_opts=None,
):
    """
    Writes a `NumPy` structured array to a HDF5 `compound` type
    dataset.

    :param data:
        The `NumPy` structured array containing the data to be
        written to disk.

    :param dset_name:
        A `str` containing the name and location of the dataset
        to write to.

    :param group:
        A h5py `Group` or `File` object from which to write the
        dataset to.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param title:
        A `str` containing the title name of the `Table` dataset.
        Default is 'Table'.

    :param attrs:
        A `dict` of key, value items to be attached as attributes
        to the `Table` dataset.

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.
    """
    kwargs = compression.settings(filter_opts)
    dset = group.create_dataset(dset_name, data=data, **kwargs)
    attach_table_attributes(dset, title, attrs)


def write_dataframe(
    df,
    dset_name,
    group,
    compression=H5CompressionFilter.LZF,
    title="Table",
    attrs=None,
    filter_opts=None,
):
    """
    Converts a `pandas.DataFrame` to a HDF5 `Table`, stored
    internall as a compound datatype.

    :param df:
        A `pandas.DataFrame` object.

    :param dset_name:
        A `str` containing the name and location of the dataset
        to write to.

    :param group:
        A h5py `Group` or `File` object from which to write the
        dataset to.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param title:
        A `str` containing the title name of the `Table` dataset.
        Default is 'Table'.

    :param attrs:
        A `dict` of key, value items to be attached as attributes
        to the `Table` dataset.

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.
    """
    # get the name and datatypes for the indices, and columns
    # check for object types, for now write fixed length strings,
    # datetime objects can come later
    # None names will be converted to 'level_{n}' for n in range(names)
    # Ensure that any numeric name is converted to bytes
    dtype = []
    idx_names = []
    default_label = "level_{}"
    dtype_metadata = {}

    # index datatypes
    for i, idx_name in enumerate(df.index.names):
        idx_data = df.index.get_level_values(i)
        if idx_name is None and len(df.index.names) == 1:
            idx_name = "index"
        elif idx_name is None:
            idx_name = default_label.format(i)
        idx_names.append(idx_name)
        _dtype_name = idx_data.dtype.name
        if "datetime64" in _dtype_name:
            # Pandas supports timezones but numpy does not; remove timezone
            _dtype_name = re.sub(r"\[ns(?:,[^\]]+)?\]", "[ns]", _dtype_name)
        dtype_metadata["{}_dtype".format(idx_name)] = _dtype_name
        if _dtype_name == "object":
            # sixy6e; there is probably a better way to do this now
            dtype.append((idx_name, VLEN_STRING))

            # fix for handling strings post h5py-3.0.0
            # converting to a numpy array to get the fixed length unicode
            # and inserting the datatype as additional metadata for deserialisation
            dtype_metadata["{}_dtype".format(idx_name)] = str(
                idx_data.values.astype("U").dtype
            )
        elif "datetime64" in _dtype_name:
            dtype.append((idx_name, "int64"))
        else:
            dtype.append((idx_name, idx_data.dtype))

    # column datatypes
    for i, val in enumerate(df.dtypes):
        col_name = df.columns[i]
        _dtype_name = val.name
        if "datetime64" in _dtype_name:
            # Pandas supports timezones but numpy does not; remove timezone
            _dtype_name = re.sub(r"\[ns(?:,[^\]]+)?\]", "[ns]", _dtype_name)
        dtype_metadata["{}_dtype".format(col_name)] = _dtype_name
        if _dtype_name == "object":
            dtype.append((col_name, VLEN_STRING))

            # metadata injection to resolve deserialisation of strings post h5py-3.0.0
            dtype_metadata["{}_dtype".format(col_name)] = str(
                df[col_name].values.astype("U").dtype
            )
        elif ("datetime64" in _dtype_name) or ("timedelta64" in _dtype_name):
            dtype.append((col_name, "int64"))
        else:
            dtype.append((col_name, val))

    dtype = numpy.dtype(dtype)

    kwargs = compression.settings(filter_opts)
    kwargs["shape"] = (df.shape[0],)
    kwargs["dtype"] = dtype
    dset = group.create_dataset(dset_name, **kwargs)

    # write the data
    # index data
    for i, idx_name in enumerate(idx_names):
        data = df.index.get_level_values(i).values
        # NumPy guards against viewing object arrays
        try:
            dset[idx_name] = data
        except (ValueError, TypeError):
            # forced to make a copies
            dset[idx_name] = data.astype("S").astype([(idx_name, VLEN_STRING)])

    # column data
    for col in df.columns:
        data = df[col].values
        # NumPy guards against viewing object arrays
        try:
            dset[col] = data
        except (ValueError, TypeError):
            # forced to make a copies
            dset[col] = data.astype("S").astype([(col, VLEN_STRING)])

    # make a copy so as not to modify the users data
    attributes = {} if attrs is None else attrs.copy()

    # insert some basic metadata
    attributes["index_names"] = numpy.array(idx_names, VLEN_STRING)
    attributes["metadata"] = (
        "`Pandas.DataFrame` converted to HDF5 compound " "datatype."
    )
    attributes["nrows"] = df.shape[0]
    attributes["python_type"] = "`Pandas.DataFrame`"
    for key in dtype_metadata:
        attributes[key] = dtype_metadata[key]
    attach_table_attributes(dset, title=title, attrs=attributes)


def read_h5_table(fid, dataset_name, dataframe=True):
    """
    Read a HDF5 `TABLE` as a `pandas.DataFrame`.

    :param fid:
        A h5py `Group` or `File` object from which to read the
        dataset from.

    :param dataset_name:
        A `str` containing the pathname of the dataset location.

    :param dataframe:
        A `bool` indicating whether to return as a `pandas.DataFrame`
        or as NumPy structured array. Default is True
        which is to return as a `pandas.DataFrame`.

    :return:
        Either a `pandas.DataFrame` (Default) or a NumPy structured
        array.
    """

    dset = fid[dataset_name]
    idx_names = None

    # grab the index names if we have them
    idx_names = dset.attrs.get("index_names")

    if dataframe:
        if dset.attrs.get("python_type") == "`Pandas.DataFrame`":
            col_names = dset.dtype.names
            dtypes = [dset.attrs["{}_dtype".format(name)] for name in col_names]
            dtype = numpy.dtype(list(zip(col_names, dtypes)))
            data = pandas.DataFrame.from_records(dset[:].astype(dtype), index=idx_names)
        else:
            data = pandas.DataFrame.from_records(dset[:], index=idx_names)
    else:
        data = dset[:]

    return data


def attach_attributes(dataset, attrs=None):
    """
    A small utility for attaching attributes to a h5py `Dataset` or
    `Group` object.

    :param dataset:
        The h5py `Dataset` or `Group` object on which to attach
        attributes to.

    :param attrs:
        A `dict` of key, value pairs used to attach the atrrbutes
        onto the h5py `Dataset` or `Group` object.

    :return:
        None.
    """
    if attrs is not None:
        for key in attrs:
            if isinstance(attrs[key], datetime.datetime):
                attrs[key] = attrs[key].isoformat()
            dataset.attrs[key] = attrs[key]


def write_scalar(data, dataset_name, group, attrs=None):
    """
    Creates a `scalar` dataset of the name given by `dataset_name`
    attached to `group`.

    :param data:
        Contains any single valued datatype.

    :param dataset_name:
        A `str` containing the name and location of the dataset
        to write to.

    :param group:
        A h5py `Group` or `File` object from which to write the
        dataset to.

    :param attrs:
        A `dict` of key, value pairs used to attach the atrrbutes
        onto the h5py `Dataset` or `Group` object.

    :return:
        None.
    """
    dset = group.create_dataset(dataset_name, data=data)
    attach_attributes(dset, attrs=DEFAULT_SCALAR_CLASS)
    attach_attributes(dset, attrs=attrs)


def h5ls(h5_obj, verbose=False):
    """
    Given an h5py `Group`, `File` (opened file id; fid), or `Dataset`,
    recursively print the contents of the HDF5 file.

    :param h5_obj:
        A h5py `Group`, `File` or `Dataset` object to use as the
        entry point from which to start listing the contents.

    :param verbose:
        If set to True, then print the attributes of each Group and
        Dataset. Default is False.
    """

    def custom_print(path):
        """
        A custom print function for dealing with HDF5 object
        types, and print formatting.
        """
        try:
            pathname = normpath(ppjoin("/", path.decode("utf-8")))
        except AttributeError:
            pathname = normpath(ppjoin("/", path))

        obj = h5_obj[path]
        attrs = {k: v for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Group):
            h5_type = "`Group`"
        elif isinstance(obj, h5py.Dataset):
            class_name = obj.attrs.get("CLASS")
            h5_class = "" if class_name is None else class_name
            fmt = "`{}Dataset`" if class_name is None else "`{} Dataset`"
            h5_type = fmt.format(h5_class)
        else:
            h5_type = "`Other`"  # we'll deal with links and references later

        print("{path}\t{h5_type}".format(path=pathname, h5_type=h5_type))
        if verbose:
            print("Attributes:")
            pprint(attrs, width=100)
            print("*" * 80)

    if isinstance(h5_obj, h5py.Dataset):
        custom_print(h5_obj.name)
    else:
        root = h5py.h5g.open(h5_obj.id, b".")
        root.links.visit(custom_print)


def read_scalar(group, dataset_name):
    """
    Read a HDF5 `SCALAR` as a dict.
    All attributes will be assigned as key: value pairs, and the
    scalar value will be assigned the key name 'value'.

    :param group:
        A h5py `Group` or `File` object from which to write the
        dataset to.

    :param dataset_name:
        A `str` containing the pathname of the dataset location.

    :return:
        A `dict` containing the SCALAR value as well as any
        attributes coupled with the SCALAR dataset.
    """
    dataset = group[dataset_name]
    data = {k: v for k, v in dataset.attrs.items()}
    data["value"] = dataset[()]
    return data


def find(h5_obj, dataset_class=""):
    """
    Given an h5py `Group`, `File` (opened file id; fid),
    recursively list all objects or optionally only list
    `h5py.Dataset` objects matching a given class, for example:

        * IMAGE
        * TABLE
        * SCALAR

    :param h5_obj:
        A h5py `Group` or `File` object to use as the
        entry point from which to start listing the contents.

    :param dataset_class:
        A `str` containing a CLASS name identifier, eg:

        * IMAGE
        * TABLE
        * SCALAR

        Default is an empty string `''`.

    :return:
        A `list` containing the pathname to all matching objects.
    """

    def _find(items, dataset_class, name, obj):
        """
        An internal utility to find objects matching `dataset_class`.
        """
        if obj.attrs.get("CLASS") == dataset_class:
            items.append(name)

    items = []
    h5_obj.visititems(partial(_find, items, dataset_class))

    return items

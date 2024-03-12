#!/usr/bin/env python

"""Various interpolation methods."""

import logging
import math

import numexpr
import numpy as np

from wagl.constants import DatasetName, GroupName, Method, Workflow
from wagl.hdf5 import (
    H5CompressionFilter,
    read_h5_table,
    write_h5_image,
)

DEFAULT_ORIGIN = (0, 0)
DEFAULT_SHAPE = (8, 8)

_LOG = logging.getLogger(__name__)


def bilinear(shape, fUL, fUR, fLR, fLL, dtype=np.float64):
    """Bilinear interpolation of four scalar values.

    :param shape:
        Shape of interpolated grid (nrows, ncols).

    :param fUL:
        Data value at upper-left (NW) corner.

    :param fUR:
        Data value at upper-right (NE) corner.

    :param fLR:
        Data value at lower-right (SE) corner.

    :param fLL:
        Data value at lower-left (SW) corner.

    :param dtype:
        Data type (numpy I presume?).

    :return:
        Array of data values interpolated between corners.
    """
    s, t = (a.astype(dtype) for a in np.ogrid[0 : shape[0], 0 : shape[1]])

    s /= shape[0] - 1.0
    t /= shape[1] - 1.0

    result = s * (t * fLR + (1.0 - t) * fLL) + (1.0 - s) * (t * fUR + (1.0 - t) * fUL)

    return result


def indices(origin=DEFAULT_ORIGIN, shape=DEFAULT_SHAPE):
    """Generate corner indices for a grid block.

    :param origin:
        Block origin (2-tuple).

    :param shape:
        Block shape (2-tuple: nrows, ncols).

    :return:
        Corner indices: (xmin, xmax, ymin, ymax).
    """
    return (origin[0], origin[0] + shape[0] - 1, origin[1], origin[1] + shape[1] - 1)


def subdivide(origin=DEFAULT_ORIGIN, shape=DEFAULT_SHAPE):
    """Generate indices for grid sub-blocks.

    :param origin:
        Block origin (2-tuple).

    :param shape:
        Block shape (nrows, ncols).

    :return:
        Dictionary containing sub-block corner indices:
            { 'UL': <list of 2-tuples>,
              'UR': <list of 2-tuples>,
              'LL': <list of 2-tuples>,
              'LR': <list of 2-tuples> }
    """
    i0, ie, j0, je = indices(origin, shape)
    ic = origin[0] + shape[0] // 2
    jc = origin[1] + shape[1] // 2

    return {
        "UL": [(i0, j0), (i0, jc), (ic, j0), (ic, jc)],
        "LL": [(ic, j0), (ic, jc), (ie, j0), (ie, jc)],
        "UR": [(i0, jc), (i0, je), (ic, jc), (ic, je)],
        "LR": [(ic, jc), (ic, je), (ie, jc), (ie, je)],
    }


def interpolate_block(
    origin=DEFAULT_ORIGIN, shape=DEFAULT_SHAPE, eval_func=None, grid=None
):
    """Interpolate a grid block.

    :param origin:
        Block origin (2-tuple).

    :param shape:
        Block shape (nrows, ncols).

    :param eval_func:
        Evaluator function.
    :type eval_func:
        callable; accepts grid indices i, j and returns a scalar value.

    :param grid:
        Grid array.
    :type grid:
        :py:class:`numpy.array`.

    :return:
        Interpolated block array if grid argument is None. If grid argument
        is supplied its elements are modified in place and this function
        does not return a value.
    """
    i0, i1, j0, j1 = indices(origin, shape)

    fUL = eval_func(i0, j0)
    fLL = eval_func(i1, j0)
    fUR = eval_func(i0, j1)
    fLR = eval_func(i1, j1)

    if grid is None:
        return bilinear(shape, fUL, fUR, fLR, fLL)

    grid[i0 : i1 + 1, j0 : j1 + 1] = bilinear(shape, fUL, fUR, fLR, fLL)


def interpolate_grid(
    grid, eval_func, depth=0, origin=DEFAULT_ORIGIN, shape=DEFAULT_SHAPE
):
    """Entry function for recursive inplace grid interpolation.

    :param grid:
        Grid array.
    :type grid:
        :py:class:`numpy.array`.

    :param eval_func:
        Evaluator function.
    :type eval_func:
        callable; accepts grid indices i, j and returns a scalar value.

    :param depth:
        Recursive bisection depth.
    :type depth:
        :py:class:`int`

    :param origin:
        Block origin,
    :type origin:
        :py:class:`tuple` of length 2.

    :param shape:
        Block shape.
    :type shape:
        :py:class:`tuple` of length 2 ``(nrows, ncols)``.
    """
    # bilinear requires a 2 by 2 grid at a minimum;
    #  depth can be derived by bit length
    max_depth = min(shape[0].bit_length(), shape[1].bit_length()) - 2
    if max_depth < 0:
        raise ValueError("Unable to interpolate grid of %s" % str(shape))
    elif max_depth < depth:
        _LOG.warning(
            f"Requested depth of {depth} but maximum interpolated depth is {max_depth};"
            f" using {max_depth} for shape {shape!s}"
        )
        depth = max_depth
    return __interpolate_grid_inner(grid, eval_func, depth, origin, shape)


def __interpolate_grid_inner(grid, eval_func, depth, origin, shape):
    """Recursive calls to interpolate a gridded dataset
    Interpolation is performed inplace to the provided grid.

    :param grid:
        Grid array.
    :type grid:
        :py:class:`numpy.array`.

    :param eval_func:
        Evaluator function.
    :type eval_func:
        callable; accepts grid indices i, j and returns a scalar value.

    :param depth:
        Recursive bisection depth.
    :type depth:
        :py:class:`int`

    :param origin:
        Block origin,
    :type origin:
        :py:class:`tuple` of length 2.

    :param shape:
        Block shape.
    :type shape:
        :py:class:`tuple` of length 2 ``(nrows, ncols)``.
    """
    if depth == 0:
        interpolate_block(origin, shape, eval_func, grid)
    else:
        blocks = subdivide(origin, shape)
        for kUL, _, _, kLR in blocks.values():
            block_shape = (kLR[0] - kUL[0] + 1, kLR[1] - kUL[1] + 1)
            __interpolate_grid_inner(grid, eval_func, depth - 1, kUL, block_shape)


def sheared_bilinear_interpolate(
    cols,
    rows,
    locations,
    samples,
    row_start,
    row_end,
    row_centre,
    shear=True,
    both_sides=False,
):
    """Generalisation of the original NBAR interpolation scheme.

    -   locations/samples may be greater than 9 (e.g. 25, 49, etc)
    -   two additional configuation options:

    :bool shear:
        If false then apply textbook bilinear interpolation. Note this
        expects that the locations describe a rectilinear grid. If true
        then make adjustments for grid distortion and curvature of
        boxlines (row_start, row_centre and row_end).
        See also `both_sides`.

    :bool both_sides:
        Only has effect if shear is True.
        If false then apply original modification like used in the
        previous fortran version.
        If true then instead apply corrections for trapezoidal shape of
        4-point grid cells.
        If the grid/boxlines have a sheared parallelogram shape (rather
        than more general sheared trapezoidal shapes) then both
        methods should produce equivalent output, otherwise the
        original version is expected to introduce larger discontinuities
        between cells.

    Optimised to reduce memory footprint (no large rasters are
    temporarily allocated).
    """
    # pylint: disable=unused-variable

    n = len(samples)
    grid_size = int(math.sqrt(n)) - 1

    assert (grid_size + 1) ** 2 == n
    assert not grid_size % 2
    # Assume count of samples is 9 or 25, 49, 81.. (Grid size is 2, 4, 6, ..)

    # facilitate indexing
    locations = locations.reshape((grid_size + 1, grid_size + 1, 2))
    samples = samples.reshape((grid_size + 1,) * 2)

    # BOXLINE:
    # Parcel boundaries follow satellite track (by 1D linear interpolation)

    lines = np.empty((grid_size + 1, rows), dtype=np.uint32)

    middle_vertex = grid_size // 2

    lines[0] = row_start
    lines[middle_vertex] = row_centre
    lines[-1] = row_end

    for i in range(1, middle_vertex):
        lines[i] = row_start + (row_centre - row_start) * (i / middle_vertex)
        lines[i + middle_vertex] = row_centre + (row_end - row_centre) * (
            i / middle_vertex
        )

    # enable broadcast
    lines = lines.reshape(grid_size + 1, rows, 1)
    row_start = row_start[:, None]

    # Generate coordinate arrays
    y, x = np.ogrid[:rows, :cols]

    # Declare output raster (filled with NaN)
    result = np.full((rows, cols), np.nan, dtype=np.float32)

    # Loop over all grid cells
    for i in range(grid_size):
        lower, upper = locations[i : i + 2, 0, 0]
        for j in range(grid_size):
            left, right = lines[j : j + 2]

            values = samples[i : i + 2, j : j + 2].reshape(4)
            vertices = (
                locations[i : i + 2, j : j + 2]
                .reshape(4, 2)
                .astype(np.float32, copy=True)
            )
            # note, copying permits modification by shear

            # build numexpr to update cell with interpolation

            subset = "((left <= x) & (x <= right)) & " "((lower <= y) & (y <= upper))"

            exp = "a0 + a1*y + a2*x + a3*x*y"

            # apply shear, to warp this interpolation

            if shear:
                if not both_sides:
                    sheared = "x - row_start"
                else:
                    # if near-singular matrix warnings, multiply by a constant typical width-between-samples  # noqa: E501
                    sheared = "(x - left) / (right - left)"

                # apply shear to interpolation
                exp = exp.replace("x", "(" + sheared + ")")

                # retrieve original y,x coordinates (i.e. indices) for 4 vertices
                vi, vj = map(list, vertices.T.astype(int))

                # Update vertices with same warp as for the interpolation
                four_pts = {
                    "x": x[:, vj].ravel(),
                    "left": left[vi].ravel(),
                    "right": right[vi].ravel(),
                    "row_start": row_start[vi].ravel(),
                }
                vertices[:, 1] = numexpr.evaluate(sheared, local_dict=four_pts)

            # determine bilinear coefficients

            matrix = np.ones((4, 4))
            matrix[:, 1:3] = vertices
            matrix[:, 3] = vertices[:, 0] * vertices[:, 1]

            # pylint: disable=unused-variable
            a0, a1, a2, a3 = np.linalg.solve(matrix, values)

            # update output raster

            expression = f"where({subset}, {exp}, result)"

            numexpr.evaluate(expression, out=result, casting="same_kind")

    return result


def interpolate(
    acq,
    coefficient,
    ancillary_group,
    satellite_solar_group,
    coefficients_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    method=Method.SHEARB,
):
    # TODO: more docstrings
    """Perform interpolation."""
    if method not in Method:
        msg = "Interpolation method {} not available."
        raise Exception(msg.format(method.name))

    geobox = acq.gridded_geo_box()
    cols, rows = geobox.get_shape_xy()

    # read the relevant tables into DataFrames
    coordinator = read_h5_table(ancillary_group, DatasetName.COORDINATOR.value)
    boxline = read_h5_table(satellite_solar_group, DatasetName.BOXLINE.value)

    if coefficient in Workflow.NBAR.atmos_coefficients:
        dataset_name = DatasetName.NBAR_COEFFICIENTS.value
    elif coefficient in Workflow.SBT.atmos_coefficients:
        dataset_name = DatasetName.SBT_COEFFICIENTS.value
    else:
        msg = "Factor name not found in available coefficients: {}"
        raise ValueError(msg.format(Workflow.STANDARD.atmos_coefficients))

    coefficients = read_h5_table(coefficients_group, dataset_name)
    coord = np.zeros((coordinator.shape[0], 2), dtype="int")
    map_x = coordinator.map_x.values
    map_y = coordinator.map_y.values
    coord[:, 1], coord[:, 0] = (map_x, map_y) * ~geobox.transform
    centre = boxline.bisection_index.values
    start = boxline.start_index.values
    end = boxline.end_index.values

    band_records = coefficients.band_name == acq.band_name
    samples = coefficients[coefficient.value][band_records].values

    func_map = {
        Method.BILINEAR: sheared_bilinear_interpolate,
        Method.SHEAR: sheared_bilinear_interpolate,
        Method.SHEARB: sheared_bilinear_interpolate,
    }

    args = [cols, rows, coord, samples, start, end, centre]
    if method == Method.BILINEAR:
        args.extend([False, False])
    elif method == Method.SHEARB:
        args.extend([True, True])
    else:
        pass

    result = func_map[method](*args)

    assert out_group is not None
    fid = out_group

    if GroupName.INTERP_GROUP.value not in fid:
        fid.create_group(GroupName.INTERP_GROUP.value)

    if filter_opts is None:
        filter_opts = {}
    else:
        filter_opts = filter_opts.copy()
    filter_opts["chunks"] = acq.tile_size

    group = fid[GroupName.INTERP_GROUP.value]

    fmt = DatasetName.INTERPOLATION_FMT.value
    dset_name = fmt.format(coefficient=coefficient.value, band_name=acq.band_name)
    no_data = np.nan
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
        "interpolation_method": method.name,
        "band_id": acq.band_id,
        "band_name": acq.band_name,
        "alias": acq.alias,
        "coefficient": coefficient.value,
    }
    desc = (
        "Contains the interpolated result of coefficient {} "
        "for band {} from sensor {}."
    )
    attrs["description"] = desc.format(coefficient.value, acq.band_id, acq.sensor_id)

    result[result == -999] = no_data
    write_h5_image(result, dset_name, group, compression, attrs, filter_opts)

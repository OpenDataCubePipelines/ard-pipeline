
#    config = Configuration('wagl', parent_package, top_path)
#    config.add_subpackage('scripts')



# These are included via utils/
#    config.add_data_files('acquisition/sensors.json')
#    config.add_data_dir('spectral_response')


incdir_numpy = run_command(
    py3,
    [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
    ],
    check: true,
).stdout().strip()
incdir_f2py = run_command(
    py3,
    [
        '-c',
        'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
    ],
    check: true,
).stdout().strip()
inc_np = include_directories(incdir_numpy, incdir_f2py)

# Share this object across multiple modules.
fortranobject_lib = static_library(
    '_fortranobject',
    incdir_f2py / 'fortranobject.c',
    # c_args: numpy_nodepr_api,
    dependencies: py3_dep,
    include_directories: [inc_np],
)
fortranobject_dep = declare_dependency(
    link_with: fortranobject_lib,
    include_directories: [inc_np],
)

sat_sol_angles_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/solar_angle.f90',
    'f90_sources/geod2geo.f90',
    'f90_sources/q_cal.f90',
    'f90_sources/compute_angles.f90',
    'f90_sources/satellite_solar_angles_main.f90',
]
_sat_sol_angles_module = custom_target(
    'sat_sol_angles_module',
    input: sat_sol_angles_src,
    output: [
        '__sat_sol_anglesmodule.c',
        '__sat_sol_angles-f2pywrappers2.f90',
        '__sat_sol_angles-f2pywrappers.f',
    ],
    command: [
        'f2py',
        '-m', '__sat_sol_angles',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_sat_sol_angles = py3.extension_module(
    '__sat_sol_angles',
    [
        _sat_sol_angles_module,
        sat_sol_angles_src,
    ],
    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)


cast_shadow_mask_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/cast_shadow_main.f90',
    'f90_sources/terrain_border_margins.f90',
    'f90_sources/cast_shadow_mask.f90',
    'f90_sources/terrain_occlusion.f90',
    'f90_sources/geo2metres_pixel_size.f90',
]
_cast_shadow_mask_module = custom_target(
    'cast_shadow_mask_module',
    input: cast_shadow_mask_src,
    output: [
        '__cast_shadow_maskmodule.c',
        '__cast_shadow_mask-f2pywrappers.f',
        '__cast_shadow_mask-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__cast_shadow_mask',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_cast_shadow_mask = py3.extension_module(
    '__cast_shadow_mask',
    [
        _cast_shadow_mask_module,
        cast_shadow_mask_src,
    ],
    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)

exiting_angle_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/exiting_angle.f90',
    'f90_sources/earth_rotation.f90',
]
_exiting_angle_module = custom_target(
    'exiting_angle_module',
    input: exiting_angle_src,
    output: [
        '__exiting_anglemodule.c',
        '__exiting_angle-f2pywrappers.f',
        '__exiting_angle-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__exiting_angle',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_exiting_angle = py3.extension_module(
    '__exiting_angle',
    [
        _exiting_angle_module,
        exiting_angle_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)
incident_angle_src = [
    'f90_sources/incident_angle.f90',
    'f90_sources/earth_rotation.f90',
]
_incident_angle_module = custom_target(
    'incident_angle_module',
    input: incident_angle_src,
    output: [
        '__incident_anglemodule.c',
        '__incident_angle-f2pywrappers.f',
        # '__incident_angle-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__incident_angle',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_incident_angle = py3.extension_module(
    '__incident_angle',
    [
        _incident_angle_module,
        incident_angle_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)
slope_aspect_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/slope_aspect.f90',
    'f90_sources/geo2metres_pixel_size.f90',
]
_slope_aspect_module = custom_target(
    'slope_aspect_module',
    input: slope_aspect_src,
    output: [
        '__slope_aspectmodule.c',
        '__slope_aspect-f2pywrappers.f',
        '__slope_aspect-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__slope_aspect',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_slope_aspect = py3.extension_module(
    '__slope_aspect',
    [
        _slope_aspect_module,
        slope_aspect_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)
surface_reflectance_src = [
    'f90_sources/surface_reflectance.f90',
    'f90_sources/white_sky.f90',
    'f90_sources/black_sky.f90',
    'f90_sources/brdf_shape.f90',
]
_surface_reflectance_module = custom_target(
    'surface_reflectance_module',
    input: surface_reflectance_src,
    output: [
        '__surface_reflectancemodule.c',
        '__surface_reflectance-f2pywrappers.f',
        # '__surface_reflectance-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__surface_reflectance',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_surface_reflectance = py3.extension_module(
    '__surface_reflectance',
    [
        _surface_reflectance_module,
        surface_reflectance_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)
satellite_model_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/geo2metres_pixel_size.f90',
    'f90_sources/satellite_model.f90',
]
_satellite_model_module = custom_target(
    'satellite_model_module',
    input: satellite_model_src,
    output: [
        '__satellite_modelmodule.c',
        '__satellite_model-f2pywrappers.f',
        '__satellite_model-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__satellite_model',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_satellite_model = py3.extension_module(
    '__satellite_model',
    [
        _satellite_model_module,
        satellite_model_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)
track_time_info_src = [
    'f90_sources/sys_variables.f90',
    'f90_sources/geod2geo.f90',
    'f90_sources/q_cal.f90',
    'f90_sources/geo2metres_pixel_size.f90',
    'f90_sources/satellite_track.f90',
    'f90_sources/track_time_info.f90',

]
_track_time_info_module = custom_target(
    'track_time_info_module',
    input: track_time_info_src,
    output: [
        '__track_time_infomodule.c',
        '__track_time_info-f2pywrappers.f',
        '__track_time_info-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__track_time_info',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_track_time_info = py3.extension_module(
    '__track_time_info',
    [
        _track_time_info_module,
        track_time_info_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)


bilinear_interpolation_src = [
    'f90_sources/bilinear_interpolation.f90',

]
_bilinear_interpolation_module = custom_target(
    'bilinear_interpolation_module',
    input: bilinear_interpolation_src,
    output: [
        '__bilinear_interpolationmodule.c',
        '__bilinear_interpolation-f2pywrappers.f',
        # '__bilinear_interpolation-f2pywrappers2.f90',
    ],
    command: [
        'f2py',
        '-m', '__bilinear_interpolation',
        '@INPUT@',
        '--build-dir', '@OUTDIR@',
    ],
)
_bilinear_interpolation = py3.extension_module(
    '__bilinear_interpolation',
    [
        _bilinear_interpolation_module,
        bilinear_interpolation_src,
    ],

    install: true,
    dependencies: [fortranobject_dep],
    link_language: 'fortran',
    subdir: 'wagl',
)

project(
    'ard-pipeline',
    'c',
    'fortran',
    version: run_command(
        'python3',
        ['-m', 'setuptools_scm'],
        check: true,
    ).stdout().strip(),
)

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)

py3_dep = py3.dependency()
ff = meson.get_compiler('fortran')

c = run_command('utils/python-package-files.sh', check: true)
packaged_files = c.stdout().strip().split('\n')
py3.install_sources(
    packaged_files,
    preserve_path: true,
)

fortran_args = ff.get_supported_arguments(
    ff.get_supported_arguments('-Wno-argument-mismatch'),
    ff.get_supported_arguments('-Wno-conversion'),
    ff.get_supported_arguments('-Wno-maybe-uninitialized'),
    ff.get_supported_arguments('-Wno-unused-dummy-argument'),
    ff.get_supported_arguments('-Wno-unused-label'),
    ff.get_supported_arguments('-Wno-unused-variable'),
    ff.get_supported_arguments('-Wno-tabs'),
)


subdir('wagl')

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

add_project_arguments(
    '-Wno-argument-mismatch',
    '-Wno-conversion',
    '-Wno-maybe-uninitialized',
    '-Wno-unused-dummy-argument',
    '-Wno-unused-label',
    '-Wno-unused-variable',
    '-Wno-tabs',
    language: 'fortran',
)


subdir('wagl')

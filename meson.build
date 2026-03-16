project(
  'refnx',
  'c', 'cpp', 'cython',
  license: 'BSD-3',
  version: run_command(['tools/gitversion.py'], check: true).stdout().strip(),
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c17',
    'cpp_std=c++17',
  ],
)

fs = import('fs')

py3 = import('python').find_installation(pure: false)

refnx_dir = py3.get_install_dir() / 'refnx'

subdir('refnx')

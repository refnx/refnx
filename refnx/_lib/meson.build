py3.install_sources([
  '__init__.py',
  '_numdiff.py',
  '_pymc.py',
  '_testutils.py',
  'util.py',
  ],
  subdir: 'refnx/_lib'
)

py3.extension_module(
  '_cutil',
  '../../src/_cutil.pyx',
  install: true,
  subdir: 'refnx/_lib/' # Places it in my_package.submodules
)

install_subdir('emcee',
  install_dir : refnx_dir / '_lib/emcee',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__', 'moves/__pycache__', 'backends/__pycache__'],
)

install_subdir('ptemcee',
  install_dir : refnx_dir / '_lib/ptemcee',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__'],
)

subdir('tests')

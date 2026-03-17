py3.install_sources([
  '__init__.py',
  '_auto_reduction.py',
  '_tof_simulator.py',
  'batchreduction.py',
  'bounded_splines.py',
  'event.py',
  'manual_beam_finder.py',
  'parabolic_motion.py',
  'peak_utils.py',
  'platypusnexus.py',
  'rebin.py',
  'reduce.py',
  'xray.py',
  ],
  subdir: 'refnx/reduce'
)

py3.extension_module(
  '_cevent',
  '../../src/_cevent.pyx',
  install: true,
  include_directories: [inc_np],
  dependencies: [np_dep],
  subdir: 'refnx/reduce/' # Places it in my_package.submodules
)

install_subdir('tests',
  install_dir : refnx_dir / 'reduce/tests',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__'],
)

install_subdir('_app',
  install_dir : refnx_dir / 'reduce/_app',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__', 'ui/__pycache__'],
)

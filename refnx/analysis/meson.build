py3.install_sources([
  '__init__.py',
  'bounds.py',
  'curvefitter.py',
  'model.py',
  'objective.py',
  'parameter.py',
  ],
  subdir: 'refnx/analysis'
)

install_subdir('tests',
  install_dir : refnx_dir / 'analysis/tests',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__', 'NIST_STRD/__pycache__'],
)

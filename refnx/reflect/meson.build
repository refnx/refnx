py3.install_sources([
  '__init__.py',
  '_code_fragment.py',
  '_functional_form.py',
  '_creflect.pxd',
  '_functional_form.py',
  '_interactive_modeller.py',
  '_jax_reflect.py',
  '_lipid.py',
  '_numba_reflect.py',
  '_polarised_reflect_model.py',
  '_reflect.py',
  'interface.py',
  'reflect_model.py',
  'spline.py',
  'structure.py',
  ],
  subdir: 'refnx/reflect'
)

srcs = ['../../src/_creflect.pyx', '../../src/refcaller.cpp', '../../src/pnr/magnetic.cc']
if is_windows and not is_mingw
  srcs += '../../src/refcalc.cpp'
else
  srcs += '../../src/refcalc.c'
endif

py3.extension_module(
  '_creflect',
  srcs,
  install: true,
  override_options : ['cython_language=cpp'],
  include_directories: [inc_np, '../../src', '../../src/pnr'],
  dependencies: [np_dep],
  subdir: 'refnx/reflect'
)

if omp_dep.found()
    srcs2 = ['../../src/_cyreflect.pyx', '../../src/refcaller.cpp', '../../src/pnr/magnetic.cc']
    if is_windows and not is_mingw
        srcs2 += '../../src/refcalc.cpp'
    else
        srcs2 += '../../src/refcalc.c'
    endif
    py3.extension_module(
      '_cyreflect',
      srcs2,
      install: true,
      override_options : ['cython_language=cpp'],
      include_directories: [inc_np, '../../src', '../../src/pnr'],
      dependencies: [np_dep, omp_dep],
      subdir: 'refnx/reflect'
    )
endif


install_subdir('tests',
  install_dir : refnx_dir / 'reflect/tests',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__'],
)

install_subdir('_app',
  install_dir : refnx_dir / 'reflect/_app',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__', 'icons/__pycache__', 'tests/__pycache__', 'ui/__pycache__'],
)

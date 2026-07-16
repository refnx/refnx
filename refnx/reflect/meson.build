py3.install_sources([
  '__init__.py',
  '_code_fragment.py',
  '_functional_form.py',
  '_creflect.pxd',
  '_functional_form.py',
  '_interactive_modeller.py',
  '_jax_reflect.py',
  '_jax_compiler.py',
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


# Define the flags you want
safer_args = cc.get_supported_arguments([
  '-O3',
  '-march=native',
  '-fno-math-errno',      # safe: errno never checked after math calls
  '-fassociative-math',   # mostly safe: minor FP reordering for vectorisation
  '-fno-trapping-math',   # safe: no FP exception trapping needed
  # Deliberately excluded:
  # -fno-signed-zeros     UNSAFE: TINY sentinel relies on correct sign
  # -ffinite-math-only    UNSAFE: bad SLD params could produce NaN/Inf
])


py3.extension_module(
  '_creflect',
  srcs,
  install: true,
  override_options : ['cython_language=cpp'],
  include_directories: [inc_np, '../../src', '../../src/pnr'],
  dependencies: [np_dep],
  subdir: 'refnx/reflect',
  limited_api: '3.11',
  c_args: safer_args,
  cpp_args: safer_args,
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
      subdir: 'refnx/reflect',
      limited_api: '3.11'
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

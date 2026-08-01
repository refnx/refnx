py3.install_sources([
  '__init__.py',
  '_code_fragment.py',
  '_functional_form.py',
  '_creflect.pxd',
  '_functional_form.py',
  '_interactive_modeller.py',
  '_jax_reflect.py',
  '_torch_reflect.py',
  '_abeles_jax_ffi_wrapper.py',
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


# Optional: XLA FFI target wrapping the C `abeles` kernel for jax, so it can
# be called as a fast forward pass with gradients piggybacked onto jabeles
# (see refnx/reflect/_abeles_jax_ffi_wrapper.py). Only built if jax is importable in
# the build environment -- absent jax this is silently skipped, same pattern
# as the omp_dep-gated _cyreflect target above.
jax_ffi_check = run_command(
  py3, ['-c', 'import jax.ffi, sys; sys.stdout.write(jax.ffi.include_dir())'],
  check: false,
)
if jax_ffi_check.returncode() == 0
  jax_ffi_incdir = include_directories(jax_ffi_check.stdout().strip())

  abeles_ffi_srcs = ['../../src/refcalc_ffi.cpp']
  if is_windows and not is_mingw
    abeles_ffi_srcs += '../../src/refcalc.cpp'
  else
    abeles_ffi_srcs += '../../src/refcalc.c'
  endif

  py3.extension_module(
    '_abeles_jax_ffi',
    abeles_ffi_srcs,
    install: true,
    include_directories: [jax_ffi_incdir, '../../src'],
    subdir: 'refnx/reflect',
    limited_api: '3.11',
    cpp_args: safer_args,
    c_args: safer_args,
  )
else
  warning('jax not found at build time -- skipping abeles_jax_ffi target')
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

install_subdir('extra',
  install_dir : refnx_dir / 'reflect/extra',
  strip_directory : true, # Set to true if you only want contents, not the 'assets' folder itself,
  exclude_directories : ['__pycache__'],
)

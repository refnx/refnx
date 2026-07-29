/*
    refcalc_ffi.cpp

    Exposes the existing `abeles` C kernel (src/refcalc.c) as an XLA FFI
    handler so it can be registered as a jax custom call target.  This file
    contains no reflectivity maths of its own -- it is a thin adapter that
    unpacks XLA FFI buffers and forwards them to `abeles`.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, Australian Nuclear Science and Technology
Organisation

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

*/

#include <Python.h>

#include "xla/ffi/api/ffi.h"

extern "C" {
#include "refcalc.h"
}

namespace ffi = xla::ffi;

#if defined(_MSC_VER)
#define REFNX_FFI_EXPORT __declspec(dllexport)
#else
#define REFNX_FFI_EXPORT __attribute__((visibility("default")))
#endif

static ffi::Error AbelesImpl(ffi::Buffer<ffi::F64> coefP,
                              ffi::Buffer<ffi::F64> xP,
                              ffi::ResultBuffer<ffi::F64> yP) {
  int numcoefs = static_cast<int>(coefP.dimensions()[0]);
  int npoints = static_cast<int>(xP.dimensions()[0]);
  abeles(numcoefs, coefP.typed_data(), npoints, yP->typed_data(),
         xP.typed_data());
  return ffi::Error::Success();
}

// python extension modules are built with hidden symbol visibility by
// default; this forward declaration forces AbelesFFI to stay exported so
// ctypes can dlsym() it out of the compiled .so/.pyd at runtime.
extern "C" REFNX_FFI_EXPORT XLA_FFI_Error* AbelesFFI(
    XLA_FFI_CallFrame* call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AbelesFFI, AbelesImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()    // coefP
        .Arg<ffi::Buffer<ffi::F64>>()    // xP (q)
        .Ret<ffi::Buffer<ffi::F64>>());  // yP

/*
 * This is compiled as a `py3.extension_module` (not a bare shared_library)
 * purely so that meson-python's editable-install machinery resolves its
 * on-disk path the same way it does for `_creflect`/`_cyreflect` -- i.e.
 * `refnx.reflect._abeles_ffi.__file__` always points at the actual built
 * artefact, whether that's the build directory (editable install) or the
 * installed package directory (regular install). The module itself exposes
 * no Python-callable functionality; `AbelesFFI` above is loaded out of this
 * same shared object via ctypes at runtime (see _jax_abeles_ffi.py).
 */
static struct PyModuleDef _abeles_ffi_moduledef = {
    PyModuleDef_HEAD_INIT,
    "_abeles_ffi",
    nullptr,
    -1,
    nullptr,
};

extern "C" PyMODINIT_FUNC PyInit__abeles_ffi(void) {
  return PyModule_Create(&_abeles_ffi_moduledef);
}

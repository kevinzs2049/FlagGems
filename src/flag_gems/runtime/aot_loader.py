"""
Triton CPU AOT (Ahead-of-Time) kernel loader for FlagGems.

Eliminates JIT Python dispatch overhead by:
1. Pre-compiling kernel .so (no JIT compilation on first call)
2. Pre-compiling a Python C extension launcher (same mechanism as JIT's CPULauncher)
   with the kernel function pointer pre-resolved

The launcher is a Python C extension (.so) that:
- Accepts Python objects (tensors, ints) directly via PyArg_ParseTuple
- Extracts data_ptr() in C (same as JIT)
- Dispatches across grid with OMP parallelism
- No specialization/cache key computation overhead

Environment variables:
    FLAGGEMS_AOT_DIR: Directory for AOT cache (default: ~/.flaggems/aot_cache)
    FLAGGEMS_USE_AOT: Enable AOT kernel loading (default: 0)
"""

import ctypes
import hashlib
import importlib.util
import json
import logging
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_aot_dir():
    return os.environ.get(
        "FLAGGEMS_AOT_DIR",
        os.path.expanduser("~/.flaggems/aot_cache"),
    )


def _aot_enabled():
    return os.environ.get("FLAGGEMS_USE_AOT", "0") in ("1", "true", "True")


# Map Triton type strings to C type names
_TRITON_TYPE_TO_C = {
    "i1": "int32_t", "i8": "int8_t", "i16": "int16_t",
    "i32": "int32_t", "i64": "int64_t",
    "u1": "uint32_t", "u8": "uint8_t", "u16": "uint16_t",
    "u32": "uint32_t", "u64": "uint64_t",
    "fp16": "float", "bf16": "float", "fp32": "float",
    "f32": "float", "fp64": "double",
}


def _triton_type_to_c(ty: str) -> str:
    if ty.startswith("*"):
        return "void*"
    return _TRITON_TYPE_TO_C[ty]


def _format_of(ty: str) -> str:
    """Python format code for PyArg_ParseTuple."""
    if ty.startswith("*"):
        return "O"  # PyObject* (tensor)
    return {
        "float": "f", "double": "d",
        "int8_t": "b", "int16_t": "h", "int32_t": "i", "int64_t": "L",
        "uint8_t": "B", "uint16_t": "H", "uint32_t": "I", "uint64_t": "K",
    }[_triton_type_to_c(ty)]


def _extracted_type(ty: str) -> str:
    """C type for PyArg_ParseTuple destination variable."""
    if ty.startswith("*"):
        return "PyObject*"
    return _triton_type_to_c(ty)


def _build_launcher_src(kernel_so_path: str, kernel_name: str, arg_types: List[str]) -> str:
    """Generate Python C extension source that loads and dispatches an AOT kernel.

    This is structurally identical to triton's CPULauncher (make_launcher in driver.py),
    but with the kernel .so path and function name baked in at compile time.
    The key advantage over the JIT path: no specialization, no cache lookup, no
    binder() — just PyArg_ParseTuple → extract pointers → OMP grid dispatch.
    """
    # Build kernel function type
    kernel_fn_c_types = [_triton_type_to_c(ty) for ty in arg_types] + ["uint32_t"] * 6
    kernel_fn_type = ", ".join(kernel_fn_c_types)

    # Build arg declarations for grid dispatch
    arg_decls = ", ".join(
        f"{_triton_type_to_c(ty)} arg{i}"
        for i, ty in enumerate(arg_types)
        if not ty.startswith("*")
    )
    # For pointers, we'll pass void*
    all_arg_decls = ", ".join(
        f"{_triton_type_to_c(ty)} arg{i}" for i, ty in enumerate(arg_types)
    )
    kernel_call_args = ", ".join(f"arg{i}" for i in range(len(arg_types)))
    kernel_call_args_comma = kernel_call_args + ", " if kernel_call_args else ""

    # Parse format string
    args_format = "".join(_format_of(ty) for ty in arg_types)
    parse_format = "iii" + args_format  # gridX, gridY, gridZ + kernel args

    # Variable declarations
    var_decls = " ".join(
        f"{_extracted_type(ty)} _arg{i};" for i, ty in enumerate(arg_types)
    )

    # PyArg pointers
    arg_ptrs = ", ".join(f"&_arg{i}" for i in range(len(arg_types)))

    # Pointer extraction
    ptr_extracts = []
    for i, ty in enumerate(arg_types):
        if ty.startswith("*"):
            ptr_extracts.append(
                f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); "
                f"if (!ptr_info{i}.valid) return NULL; "
                f"void* arg{i} = ptr_info{i}.dev_ptr;"
            )
        else:
            ptr_extracts.append(f"{_triton_type_to_c(ty)} arg{i} = _arg{i};")
    ptr_extract_code = "\n  ".join(ptr_extracts)

    src = f'''
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

using kernel_ptr_t = void(*)({kernel_fn_type});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

// Kernel function pointer — loaded once from the pre-compiled .so
static kernel_ptr_t _kernel_fn = NULL;
static void *_kernel_lib = NULL;

static int _ensure_kernel_loaded(void) {{
  if (_kernel_fn) return 0;
  _kernel_lib = dlopen("{kernel_so_path}", RTLD_NOW | RTLD_LOCAL);
  if (!_kernel_lib) {{
    PyErr_Format(PyExc_RuntimeError, "AOT: dlopen failed: %s", dlerror());
    return -1;
  }}
  _kernel_fn = (kernel_ptr_t)dlsym(_kernel_lib, "{kernel_name}");
  if (!_kernel_fn) {{
    PyErr_Format(PyExc_RuntimeError, "AOT: dlsym(%s) failed: %s", "{kernel_name}", dlerror());
    dlclose(_kernel_lib);
    _kernel_lib = NULL;
    return -1;
  }}
  return 0;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  {var_decls}
  if(!PyArg_ParseTuple(args, "{parse_format}", &gridX, &gridY, &gridZ{", " + arg_ptrs if arg_ptrs else ""})) {{
    return NULL;
  }}

  if (_ensure_kernel_loaded() < 0) return NULL;

  // Extract data pointers from tensor objects
  {ptr_extract_code}

  // Grid dispatch with OMP
  uint32_t gX = gridX, gY = gridY, gZ = gridZ;
  uint32_t N = gX * gY * gZ;

  if (N == 1) {{
    (*_kernel_fn)({kernel_call_args_comma}0, 0, 0, 1, 1, 1);
  }} else if (N > 0) {{
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > 1) {{
      #pragma omp parallel for schedule(static)
      for (uint32_t i = 0; i < N; ++i) {{
        uint32_t z = i / (gX * gY);
        uint32_t rem = i % (gX * gY);
        uint32_t y = rem / gX;
        uint32_t x = rem % gX;
        (*_kernel_fn)({kernel_call_args_comma}x, y, z, gX, gY, gZ);
      }}
    }} else
#endif
    {{
      for (uint32_t z = 0; z < gZ; ++z)
        for (uint32_t y = 0; y < gY; ++y)
          for (uint32_t x = 0; x < gX; ++x)
            (*_kernel_fn)({kernel_call_args_comma}x, y, z, gX, gY, gZ);
    }}
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "AOT kernel launcher"}},
  {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__triton_aot_launcher",
  NULL, -1,
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_aot_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) return NULL;
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
'''
    return src


def _compile_aot_launcher(kernel_so_path: str, kernel_name: str,
                          arg_types: List[str], launcher_dir: str) -> Optional[str]:
    """Compile a Python C extension launcher for an AOT kernel.

    Returns path to compiled .so, or None on failure.
    """
    from triton.runtime.build import _build

    src = _build_launcher_src(kernel_so_path, kernel_name, arg_types)
    src_path = os.path.join(launcher_dir, "aot_launcher.cpp")
    with open(src_path, "w") as f:
        f.write(src)

    # Use triton's _build which handles include paths and flags correctly
    try:
        import triton._C
        import importlib.resources
        try:
            _triton_C_dir = importlib.resources.files(importlib.import_module("triton")).joinpath("_C")
        except AttributeError:
            _triton_C_dir = importlib.resources.path(importlib.import_module("triton"), "_C").__enter__()

        so_path = _build(
            "__triton_aot_launcher",
            src_path,
            launcher_dir,
            library_dirs=[_triton_C_dir],
            include_dirs=[],
            libraries=["stdc++", "dl"],
        )
        return so_path
    except Exception as e:
        logger.error(f"AOT launcher build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


class AOTKernel:
    """A pre-compiled Triton CPU kernel with a Python C extension launcher.

    Uses the exact same Python→C interface as Triton's JIT CPULauncher
    (PyArg_ParseTuple + data_ptr extraction + OMP grid dispatch), but with
    the kernel function pointer pre-resolved from a compiled .so file.
    """

    def __init__(self, launcher_so_path: str):
        spec = importlib.util.spec_from_file_location(
            "__triton_aot_launcher", launcher_so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._launch = mod.launch

    def __call__(self, gridX, gridY, gridZ, *args):
        """Launch kernel with the given grid and arguments.

        Args match the JIT launcher: gridX, gridY, gridZ, then kernel args
        (tensors or scalar values).
        """
        self._launch(gridX, gridY, gridZ, *args)


class AOTKernelCache:
    """Cache of pre-compiled AOT kernels with Python C extension launchers."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir or _get_aot_dir())
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded: Dict[str, AOTKernel] = {}

        self._manifest_path = self._cache_dir / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"kernels": {}, "version": 1, "arch": platform.machine()}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _make_key(self, kernel_name: str, constexprs: dict) -> str:
        parts = [kernel_name]
        for k in sorted(constexprs.keys()):
            parts.append(f"{k}={constexprs[k]}")
        key_str = ":".join(parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def compile_kernel(self, jit_fn, constexprs: dict, signature: dict,
                       attrs: Optional[dict] = None, num_warps: int = 1,
                       num_stages: int = 0) -> Optional[str]:
        """Compile a single kernel variant and its Python C extension launcher.

        Args:
            jit_fn: The @triton.jit decorated function
            constexprs: Dict of constexpr parameter names → values.
                        Includes both tl.constexpr params (BLOCK_N etc.) and
                        runtime-constant params (stride_ak=1 etc.)
            signature: Dict of ALL parameter names → type strings.
                       Constexpr params should be marked as 'constexpr'.
                       Non-constexpr params use their Triton type (e.g., '*bf16', 'i32').
            attrs: Optional divisibility attributes dict, keyed by (arg_index,)
        """
        import triton

        kernel_name = jit_fn.__name__
        cache_key = self._make_key(kernel_name, constexprs)
        kernel_so = self._cache_dir / f"{kernel_name}_{cache_key}.so"
        launcher_dir = self._cache_dir / f"{kernel_name}_{cache_key}_launcher"
        meta_path = self._cache_dir / f"{kernel_name}_{cache_key}.json"

        # Find the launcher .so if already compiled
        launcher_so = launcher_dir / "__triton_aot_launcher.so"
        if not launcher_so.exists():
            import glob
            matches = glob.glob(str(launcher_dir / "__triton_aot_launcher*.so"))
            if matches:
                launcher_so = Path(matches[0])

        if kernel_so.exists() and launcher_so.exists() and meta_path.exists():
            logger.info(f"AOT cache hit: {kernel_name} [{cache_key}]")
            return str(launcher_so)

        logger.info(f"AOT compiling: {kernel_name} [{cache_key}] constexprs={constexprs}")
        t0 = time.time()

        try:
            # Build signature: all args must be present, constexprs marked as 'constexpr'
            full_sig = {}
            for name in jit_fn.arg_names:
                if name in constexprs:
                    full_sig[name] = "constexpr"
                elif name in signature:
                    full_sig[name] = signature[name]
                else:
                    raise ValueError(f"Arg '{name}' not in signature or constexprs")

            src = triton.compiler.ASTSource(
                fn=jit_fn,
                constexprs=constexprs,
                signature=full_sig,
                attrs=attrs or {},
            )
            opts = {"num_warps": num_warps, "num_stages": num_stages}
            ccinfo = triton.compile(src, options=opts)

            so_binary = ccinfo.asm["so"]
            with open(kernel_so, "wb") as f:
                f.write(so_binary)

            # Build runtime arg list (non-constexpr args only)
            arg_names_runtime = []
            arg_types_runtime = []
            for name in jit_fn.arg_names:
                if name not in constexprs:
                    arg_names_runtime.append(name)
                    arg_types_runtime.append(signature[name])

            # Compile Python C extension launcher
            launcher_dir.mkdir(parents=True, exist_ok=True)
            result = _compile_aot_launcher(
                kernel_so_path=str(kernel_so),
                kernel_name=kernel_name,
                arg_types=arg_types_runtime,
                launcher_dir=str(launcher_dir),
            )
            if result is None:
                kernel_so.unlink(missing_ok=True)
                return None

            launcher_so = Path(result)

            meta = {
                "kernel_name": kernel_name,
                "cache_key": cache_key,
                "constexprs": {str(k): str(v) for k, v in constexprs.items()},
                "arg_names": arg_names_runtime,
                "arg_types": arg_types_runtime,
                "kernel_so": str(kernel_so),
                "launcher_so": str(launcher_so),
                "compiled_at": time.time(),
                "arch": platform.machine(),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            elapsed = time.time() - t0
            logger.info(f"AOT compiled: {kernel_name} [{cache_key}] in {elapsed:.1f}s")

            if kernel_name not in self._manifest["kernels"]:
                self._manifest["kernels"][kernel_name] = {}
            self._manifest["kernels"][kernel_name][cache_key] = {
                "constexprs": {str(k): str(v) for k, v in constexprs.items()},
                "launcher_so": str(launcher_so),
            }
            self._save_manifest()

            return str(launcher_so)

        except Exception as e:
            logger.error(f"AOT compilation failed for {kernel_name} [{cache_key}]: {e}")
            import traceback
            traceback.print_exc()
            kernel_so.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def load_kernel(self, kernel_name: str, constexprs: dict) -> Optional[AOTKernel]:
        """Load a pre-compiled AOT kernel."""
        cache_key = self._make_key(kernel_name, constexprs)

        if cache_key in self._loaded:
            return self._loaded[cache_key]

        meta_path = self._cache_dir / f"{kernel_name}_{cache_key}.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            launcher_so = meta["launcher_so"]
            if not os.path.exists(launcher_so):
                return None

            aot = AOTKernel(launcher_so_path=launcher_so)
            self._loaded[cache_key] = aot
            return aot
        except Exception as e:
            logger.error(f"Failed to load AOT kernel {kernel_name} [{cache_key}]: {e}")
            return None

    def compile_all_variants(self, jit_fn, variants: List[dict], signature: dict,
                             attrs: Optional[dict] = None):
        for constexprs in variants:
            self.compile_kernel(jit_fn, constexprs, signature, attrs)

    def list_cached(self) -> dict:
        return self._manifest.get("kernels", {})


# Global cache
_global_cache: Optional[AOTKernelCache] = None


def get_global_cache() -> AOTKernelCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = AOTKernelCache()
    return _global_cache

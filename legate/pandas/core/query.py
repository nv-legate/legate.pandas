# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ast
import re

import numba
import numba.cuda
import numpy as np
import six

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode

from .pattern import Map

# Utility classes to parse and execute queries using Numba.
# A great deal of code here is borrowed from cuDF.


class QueryExecutor(object):
    def __init__(self, runtime, expr, callenv, all_names, frame):
        self._runtime = runtime
        self._expr = expr
        self._callenv = callenv
        self._all_names = all_names
        self._frame = frame

        self._colnames = None
        self._refnames = None

        self._cols = None
        self._refs = None
        self._col_types = None
        self._ref_types = None

        self._mask = None

        self._func = None
        self._device_func = None

        self._use_gpu = runtime.has_gpus

    def _parse_expr(self):
        expr = self._expr.replace("@", _EXTERNAL_REFERENCE_PREFIX)
        tree = ast.parse(expr)
        _check_error(tree)
        [body] = tree.body
        extractor = _NameExtractor()
        extractor.visit(body)
        colnames = sorted(extractor.colnames)
        refnames = sorted(extractor.refnames)
        return (expr, colnames, refnames)

    @staticmethod
    def _get_numba_types(names, values, need_pointer=True):
        types = []
        for name, value in zip(names, values):
            ty = value.dtype
            ty = str(ty) if ty != bool else "int8"
            ty = getattr(numba.types, ty)
            if need_pointer:
                ty = numba.types.CPointer(ty)
            types.append(ty)
        return types

    def _prepare_args(self):
        cols = []
        # Retrieve all columns by their names
        idxr = self._all_names.get_indexer_for(self._colnames)
        if any(idx == -1 for idx in idxr):
            missing_column = self._colnames[list(idxr).index(-1)]
            raise QuerySyntaxError(f"column {missing_column} does not exist")
        cols = self._frame.select_columns(idxr)

        # Capture values in the environment and convert them to futures
        refs = []
        for ref in self._refnames:
            ref = ref.replace(_EXTERNAL_REFERENCE_PREFIX, "")
            if ref in self._callenv["locals"]:
                ref_val = self._callenv["locals"][ref]
            elif ref in self._callenv["globals"]:
                ref_val = self._callenv["globals"][ref]
            else:
                raise QuerySyntaxError("variable %s does not exist" % ref)

            dtype = np.dtype(type(ref_val))
            refs.append(
                self._runtime.create_future(ref_val, ty.to_legate_dtype(dtype))
            )
            del ref_val

        col_types = self._get_numba_types(self._colnames, cols)
        ref_types = self._get_numba_types(
            self._refnames, refs, not self._use_gpu
        )

        mask = self._runtime.create_storage(cols[0].ispace).create_column(
            ty.bool,
            ipart=cols[0].primary_ipart,
            nullable=any(c.nullable for c in cols),
        )

        return cols, refs, col_types, ref_types, mask

    def _build_cpu_func(self):
        funcid = "query_{}".format(np.uintp(hash(self._expr)))

        # Preamble
        lines = ["from numba import carray, types"]

        # Signature
        lines.append("def {}({}, {}):".format(funcid, _ARGS_VAR, _SIZE_VAR))

        # Unpack kernel arguments
        def _emit_assignment(var, idx, sz, ty):
            lines.append(
                "    {} = carray({}[{}], {}, types.{})".format(
                    var, _ARGS_VAR, idx, sz, ty
                )
            )

        arg_idx = 1
        _emit_assignment(_MASK_VAR, 0, _SIZE_VAR, numba.types.int8)
        for name, type in zip(self._colnames, self._col_types):
            _emit_assignment(name, arg_idx, _SIZE_VAR, type.dtype)
            arg_idx += 1
        for name, type in zip(self._refnames, self._ref_types):
            _emit_assignment(name, arg_idx, 0, type.dtype)
            arg_idx += 1

        # Main loop
        lines.append("    for {} in range({}):".format(_LOOP_VAR, _SIZE_VAR))

        colnames = set(self._colnames)

        # Kernel body
        def _lift_to_array_access(m):
            name = m.group(0)
            if name in colnames:
                return "{}[{}]".format(name, _LOOP_VAR)
            else:
                return "{}[0]".format(name)

        expr = re.sub(r"[_a-z]\w*", _lift_to_array_access, self._expr)
        lines.append("        {}[{}] = {}".format(_MASK_VAR, _LOOP_VAR, expr))

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs = {}
        six.exec_(body, glbs)
        return glbs[funcid]

    def _compile_func_cpu(self):
        sig = numba.types.void(
            numba.types.CPointer(numba.types.voidptr), numba.types.uint64
        )
        return numba.cfunc(sig)(self._func)

    def _execute_cpu(self):
        plan = Map(self._runtime, OpCode.EVAL_UDF)
        plan.add_scalar_arg(self._device_func.address, ty.uint64)
        self._mask.add_to_plan_output_only(plan)
        plan.add_scalar_arg(len(self._cols), ty.uint32)
        for col in self._cols:
            col.add_to_plan(plan, True)
        plan.add_scalar_arg(len(self._refs), ty.uint32)
        for ref in self._refs:
            plan.add_future(ref)
        plan.execute(self._mask.launch_domain)

    def _build_gpu_func(self):
        funcid = "query_{}".format(np.uintp(hash(self._expr)))

        # Preamble
        lines = ["from numba import cuda"]

        # Signature
        args = [_MASK_VAR] + self._colnames + self._refnames + [_SIZE_VAR]
        lines.append("def {}({}):".format(funcid, ",".join(args)))

        # Initialize the index variable and return immediately
        # when it exceeds the data size
        lines.append("    {} = cuda.grid(1)".format(_LOOP_VAR))
        lines.append("    if {} >= {}:".format(_LOOP_VAR, _SIZE_VAR))
        lines.append("        return")

        colnames = set(self._colnames)

        # Kernel body
        def _lift_to_array_access(m):
            name = m.group(0)
            if name in colnames:
                return "{}[{}]".format(name, _LOOP_VAR)
            else:
                return "{}".format(name)

        expr = re.sub(r"[_a-z]\w*", _lift_to_array_access, self._expr)
        lines.append("    {}[{}] = {}".format(_MASK_VAR, _LOOP_VAR, expr))

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs = {}
        six.exec_(body, glbs)
        return glbs[funcid]

    def _compile_func_gpu(self):
        arg_types = (
            [numba.types.CPointer(numba.types.int8)]
            + self._col_types
            + self._ref_types
            + [numba.types.uint64]
        )
        sig = (*arg_types,)

        cuda_arch = self._runtime.cuda_arch
        return numba.cuda.compile_ptx(self._func, sig, cc=cuda_arch)

    def _execute_gpu(self):
        # TODO: We need to memoize PTX loading
        # TODO: We may want to move this to the core package
        plan = Map(self._runtime, OpCode.LOAD_PTX)
        plan.add_future(
            self._runtime.create_future_from_string(self._device_func)
        )
        kernel_fun = plan.execute(self._mask.launch_domain)

        plan = Map(self._runtime, OpCode.EVAL_UDF)
        # This will be ignored
        plan.add_scalar_arg(0, ty.uint64)
        self._mask.add_to_plan_output_only(plan)
        plan.add_scalar_arg(len(self._cols), ty.uint32)
        for col in self._cols:
            col.add_to_plan(plan, True)
        plan.add_scalar_arg(len(self._refs), ty.uint32)
        for ref in self._refs:
            plan.add_future(ref)
        plan.add_future_map(kernel_fun)
        plan.execute(self._mask.launch_domain)

    def execute(self):
        (
            self._expr,
            self._colnames,
            self._refnames,
        ) = self._parse_expr()

        (
            self._cols,
            self._refs,
            self._col_types,
            self._ref_types,
            self._mask,
        ) = self._prepare_args()

        if self._use_gpu:
            self._func = self._build_gpu_func()
            (self._device_func, __) = self._compile_func_gpu()
            self._execute_gpu()
        else:
            self._func = self._build_cpu_func()
            self._device_func = self._compile_func_cpu()
            self._execute_cpu()

        return self._mask


_EXTERNAL_REFERENCE_PREFIX = "__extern_ref__"
_MASK_VAR = "__mask__"
_SIZE_VAR = "__size__"
_LOOP_VAR = "__i__"
_ARGS_VAR = "__args__"


class QuerySyntaxError(ValueError):
    pass


class _NameExtractor(ast.NodeVisitor):
    def __init__(self):
        self.colnames = set()
        self.refnames = set()

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise QuerySyntaxError("assignment is not allowed")

        name = node.id
        if name.startswith(_EXTERNAL_REFERENCE_PREFIX):
            self.refnames.add(name)
        else:
            self.colnames.add(name)


def _check_error(tree):
    if not isinstance(tree, ast.Module):
        raise QuerySyntaxError("top level should be of ast.Module")
    if len(tree.body) != 1:
        raise QuerySyntaxError("too many expressions")

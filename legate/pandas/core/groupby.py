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

from itertools import chain

from legate.core import Rect

from legate.pandas.common import types as ty
from legate.pandas.config import AggregationCode, GroupbyVariantCode, OpCode

from .partitioning import HashPartitioner
from .pattern import Map


class GroupbyReducer(HashPartitioner):
    def __init__(self, runtime, frame, key_indices, ops, method):
        super(GroupbyReducer, self).__init__(runtime)

        self._frame = frame
        self._key_indices = key_indices

        self._ops = ops
        self._method = method

        self._keys = None
        self._values = None

        self._radix = 1

        self._num_pieces = None
        self._launch_domain = None
        self._cspace = None

    def _prepare_launch_spec(self):
        return (
            self._keys[0].num_pieces,
            self._keys[0].launch_domain,
            self._keys[0].cspace,
        )

    def _parse_groupby_method(self):
        if self._num_pieces == 1:
            return GroupbyVariantCode.TREE
        try:
            return GroupbyVariantCode[self._method.upper()]
        except KeyError:
            raise NotImplementedError(
                "Unsupported merge method: %s" % self._method
            )

    @staticmethod
    def _maybe_convert(key):
        key_dtype = ty.ensure_valid_index_dtype(key.dtype)

        if key_dtype != key.dtype:
            return key.astype(key_dtype)
        else:
            return key

    def _prepare_keys(self):
        keys = [self._frame._columns[idx] for idx in self._key_indices]
        return [self._maybe_convert(key) for key in keys]

    def _prepare_values(self):
        values = []
        all_ops = []
        for col, ops in self._ops.items():
            values.append(self._frame._columns[col])
            all_ops.append([ty.get_aggregation_op_id(op) for op in ops])

        return (values, all_ops)

    @staticmethod
    def _nullable_output(column, agg_op):
        return column.nullable and not (
            agg_op == AggregationCode.COUNT
            or agg_op == AggregationCode.SIZE
            or agg_op == AggregationCode.ANY
            or agg_op == AggregationCode.ALL
        )

    def _create_output_column(self, storage, op, input):
        dtype = ty.get_reduction_result_type(op, input.dtype)
        nullable = self._nullable_output(input, op)
        output = storage.create_column(dtype, nullable=nullable)
        if ty.is_string_dtype(dtype):
            offset_storage = self._runtime.create_output_storage()
            char_storage = self._runtime.create_output_storage()
            output.add_child(
                offset_storage.create_column(ty.int32, nullable=False)
            )
            output.add_child(
                char_storage.create_column(ty.int8, nullable=False)
            )
            output = output.as_string_column()
        return output

    def _aggregate(self):
        output_storage = self._runtime.create_output_storage()

        output_keys = [
            output_storage.create_similar_column(key, False)
            for key in self._keys
        ]
        output_values = [
            [
                self._create_output_column(output_storage, op, column)
                for op in ops
            ]
            for column, ops in zip(self._values, self._ops)
        ]

        plan_groupby = Map(self._runtime, OpCode.GROUPBY_REDUCE)
        plan_groupby.add_scalar_arg(len(self._keys), ty.uint32)
        # self._radix == (Number of inputs per key column)
        plan_groupby.add_scalar_arg(self._radix, ty.uint32)
        for r in range(self._radix):
            proj_id = self._runtime.get_radix_functor_id(self._radix, r)
            for key in self._keys:
                key.add_to_plan(plan_groupby, True, proj=proj_id)
        for key in output_keys:
            key.add_to_plan_output_only(plan_groupby)
        plan_groupby.add_scalar_arg(len(self._values), ty.uint32)
        for ops, outputs, input in zip(self._ops, output_values, self._values):
            plan_groupby.add_scalar_arg(len(ops), ty.uint32)
            for op in ops:
                plan_groupby.add_scalar_arg(op.value, ty.int32)
            for output in outputs:
                output.add_to_plan_output_only(plan_groupby)
            # self._radix == (Number of inputs per column)
            plan_groupby.add_scalar_arg(self._radix, ty.uint32)
            for r in range(self._radix):
                proj_id = self._runtime.get_radix_functor_id(self._radix, r)
                input.add_to_plan(plan_groupby, True, proj=proj_id)
        counts = plan_groupby.execute(self._launch_domain)
        total_count = counts.cast(ty.int64).sum()

        output_storage = plan_groupby.promote_output_storage(output_storage)
        output_ipart = output_storage.default_ipart
        self._runtime.register_external_weighted_partition(
            output_ipart, counts
        )
        del plan_groupby

        return (output_keys, output_values, total_count)

    def _perform_reduction(self):
        return self._aggregate()

    def _construct_groupby_output(self):

        result = self._perform_reduction()

        if self._method == GroupbyVariantCode.HASH:
            # The input table is already partitioned so that chunks have
            # disjoint keys, so we only need a single round of reduction
            return result

        elif self._method == GroupbyVariantCode.TREE:
            # If we do tree-based reduction, we need to repeat reduction
            # rounds until we reach the root of the tree
            self._radix = self._runtime.radix
            while self._num_pieces > 1:
                (self._keys, self._values, total_count) = result

                self._num_pieces = (
                    self._num_pieces + self._radix - 1
                ) // self._radix
                self._launch_domain = Rect([self._num_pieces])
                self._cspace = self._runtime.find_or_create_color_space(
                    self._num_pieces
                )
                result = self._perform_reduction()

            return result

        else:
            assert False

    @staticmethod
    def _commutative_op(op):
        return op in (
            AggregationCode.SUM,
            AggregationCode.MIN,
            AggregationCode.MAX,
            AggregationCode.PROD,
        )

    def perform_groupby(self):

        # TODO: Here we need to make sure that all keys and values are
        #       partitioned in the same way
        self._keys = self._prepare_keys()
        (self._values, self._ops) = self._prepare_values()

        (
            self._num_pieces,
            self._launch_domain,
            self._cspace,
        ) = self._prepare_launch_spec()

        self._method = self._parse_groupby_method()

        partition_keys = []
        if self._method == GroupbyVariantCode.HASH:
            if self._frame.is_partitioned_by(self._key_indices):
                partition_keys = self._frame.partition_keys
            else:
                # TODO: Put back this local aggregation optimization,
                #       once the refactoring is done
                # if all(self._commutative_op(op) for op in self._agg_ops):
                #    (self._keys, self._values) = self._perform_reduction()
                key_indices = list(range(len(self._keys)))
                columns = self._keys + self._values
                columns = self._hash_partition(columns, key_indices)
                self._keys = columns[: len(self._keys)]
                self._values = columns[len(self._keys) :]
                partition_keys = key_indices

        (keys, values, total_count) = self._construct_groupby_output()

        values = list(chain(*values))

        if self._method == GroupbyVariantCode.TREE:
            keys = [key.repartition(self._runtime.num_pieces) for key in keys]
            values = [
                value.repartition(self._runtime.num_pieces) for value in values
            ]

        return (total_count, keys, values, partition_keys)

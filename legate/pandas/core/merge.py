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

import numpy as np
from pandas.errors import MergeError

from legate.pandas.common import types as ty, util as util
from legate.pandas.config import JoinTypeCode, JoinVariantCode, OpCode

from .index import create_index_from_columns, create_range_index
from .partitioning import HashPartitioner
from .pattern import Map


class Merger(HashPartitioner):
    def __init__(
        self,
        runtime,
        left,
        right,
        left_column_names,
        right_column_names,
        method,
        how,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        suffixes,
    ):

        super(Merger, self).__init__(runtime)

        self._left = left
        self._left.set_partition_keys(left.partition_keys)
        self._right = right
        self._right.set_partition_keys(right.partition_keys)

        self._method = method
        self._how = self._runtime.get_join_type_id(how)

        self._on = util.to_list_if_not_none(on)
        self._left_on = util.to_list_if_not_none(left_on)
        self._right_on = util.to_list_if_not_none(right_on)

        self._left_index = left_index
        self._right_index = right_index
        self._suffixes = suffixes

        self._left_column_names = left_column_names
        self._right_column_names = right_column_names

        self._left_key_indices = []
        self._right_key_indices = []
        self._common_indices = []
        self._output_common_columns_to_left = (
            self._how != JoinTypeCode.INNER
            or not (self._left_index and not self._right_index)
        )

        self._left_columns = None
        self._right_columns = None

        self._left_needs_conversion = []
        self._right_needs_conversion = []

        self._left_ipart = None
        self._right_ipart = None

        self._partition_keys = []

    def _parse_merge_method(self):
        if self._left_columns[0].num_pieces == 1:
            return JoinVariantCode.BROADCAST
        try:
            return JoinVariantCode[self._method.upper()]
        except KeyError:
            raise NotImplementedError(
                "Unsupported merge method: %s" % self._method
            )

    @staticmethod
    def _validate_columns(columns, to_lookup):
        idxr = columns.get_indexer_for(to_lookup)
        mask = idxr == -1
        if mask.any():
            raise KeyError(list(np.compress(mask, to_lookup)))
        return idxr

    def _find_join_keys(self):
        left_key_indices = None
        right_key_indices = None
        common_indices = None

        # This code was originally copied from Pandas and tailored to Legate
        # TODO: argument checking here should move to the frontend eventually
        if self._left_index and self._right_index:
            if self._left._index.nlevels != self._right._index.nlevels:
                raise MergeError("The number of index levels must be the same")

            nlevels = self._left.index.nlevels
            left_offset = self._left.num_columns()
            right_offset = self._right.num_columns()

            left_key_indices = [left_offset + i for i in range(nlevels)]
            right_key_indices = [right_offset + i for i in range(nlevels)]
            common_indices = list(zip(left_key_indices, right_key_indices))

        elif (
            self._on is None
            and self._left_on is None
            and self._right_on is None
        ):

            if self._left_index:
                raise MergeError("Must pass right_on or right_index=True")

            elif self._right_index:
                raise MergeError("Must pass left_on or left_index=True")

            else:
                # Use the common columns
                common_columns = self._left_column_names.intersection(
                    self._right_column_names
                )

                if len(common_columns) == 0:
                    raise MergeError(
                        "No common columns to perform merge on. "
                        "Merge options: left_on={lon}, right_on={ron}, "
                        "left_index={lidx}, right_index={ridx}".format(
                            lon=self._left_on,
                            ron=self._right_on,
                            lidx=self._left_index,
                            ridx=self._right_index,
                        )
                    )

                if not common_columns.is_unique:
                    raise MergeError(
                        f"Data columns not unique: {repr(common_columns)}"
                    )

                left_key_indices = self._validate_columns(
                    self._left_column_names, common_columns
                )
                right_key_indices = self._validate_columns(
                    self._right_column_names, common_columns
                )
                common_indices = list(zip(left_key_indices, right_key_indices))

        elif self._on is not None:
            if self._left_on is not None or self._right_on is not None:
                raise MergeError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )

            left_key_indices = self._validate_columns(
                self._left_column_names, self._on
            )
            right_key_indices = self._validate_columns(
                self._right_column_names, self._on
            )
            common_indices = list(zip(left_key_indices, right_key_indices))

        elif self._left_on is not None:
            left_key_indices = self._validate_columns(
                self._left_column_names, self._left_on
            )
            if self._right_index:
                if len(self._left_on) != self._right._index.nlevels:
                    raise ValueError(
                        "len(left_on) must equal the number "
                        'of levels in the index of "right"'
                    )
                nlevels = self._right.index.nlevels
                right_offset = self._right.num_columns()
                right_key_indices = [right_offset + i for i in range(nlevels)]
                common_indices = list(zip(left_key_indices, right_key_indices))
            elif self._right_on is None:
                raise MergeError("Must pass right_on or right_index=True")
            else:
                right_key_indices = self._validate_columns(
                    self._right_column_names, self._right_on
                )
                common_columns = list(set(self._left_on) & set(self._right_on))
                common_indices = list(
                    zip(
                        self._left_column_names.get_indexer_for(
                            common_columns
                        ),
                        self._right_column_names.get_indexer_for(
                            common_columns
                        ),
                    )
                )

        elif self._right_on is not None:
            right_key_indices = self._validate_columns(
                self._right_column_names, self._right_on
            )
            if self._left_index:
                if len(self._right_on) != self._left._index.nlevels:
                    raise ValueError(
                        "len(right_on) must equal the number "
                        'of levels in the index of "left"'
                    )
                nlevels = self._left.index.nlevels
                left_offset = self._left.num_columns()
                left_key_indices = [left_offset + i for i in range(nlevels)]
                common_indices = list(zip(left_key_indices, right_key_indices))
            else:
                raise MergeError("Must pass left_on or left_index=True")

        assert left_key_indices is not None
        assert right_key_indices is not None
        assert common_indices is not None

        if len(left_key_indices) != len(right_key_indices):
            raise ValueError("len(right_on) must equal len(left_on)")

        return (
            list(left_key_indices),
            list(right_key_indices),
            list(common_indices),
        )

    def _prepare_columns(self):
        left = self._left
        right = self._right

        # Copy the lists of columns as we may update them in place right below
        left_columns = left._columns.copy()
        right_columns = right._columns.copy()

        if self._left_index or self._right_index:
            left_columns += util.to_list_if_scalar(left._index.column)
            right_columns += util.to_list_if_scalar(right._index.column)

        return (left_columns, right_columns)

    @staticmethod
    def _maybe_convert(key, target_dtype):
        if ty.is_categorical_dtype(key.dtype):
            return key
        else:
            return key.astype(target_dtype)

    def _unify_key_dtypes(self):
        # Here we compare between the dtypes of the key columns and and perform
        # type conversions if they don't match.

        for lidx, ridx in zip(self._left_key_indices, self._right_key_indices):
            left = self._left_columns[lidx]
            right = self._right_columns[ridx]

            # Categorical dtypes are handled by the merge task
            if ty.is_categorical_dtype(left.dtype):
                if not ty.is_categorical_dtype(right.dtype):
                    raise ValueError(
                        f"Columns '{self._left_column_names[lidx]}' and "
                        f"'{self._right_column_names[ridx]}' are incompatible"
                    )

                # If we're not going to hash partition the table,
                # the following doesn't matter.
                if self._method != JoinVariantCode.HASH:
                    continue
                if not left.dtype._compare_categories(right.dtype):
                    self._left_needs_conversion.append(lidx)
                    self._right_needs_conversion.append(ridx)
                continue

            if left.dtype != right.dtype:
                # We need to make sure that we use the dtype of the column
                # that "survives" in the output
                if self._output_common_columns_to_left:
                    self._right_columns[ridx] = right.astype(left.dtype)
                else:
                    self._left_columns[lidx] = left.astype(right.dtype)

    def _radix_partition(self):
        def _partition_table(table, columns, keys, needs_conversion):
            reuse_partition = table.is_partitioned_by(keys)

            if len(needs_conversion) > 0:
                reuse_partition = []

            if not reuse_partition:
                out_columns = self._hash_partition(
                    columns, keys, needs_conversion
                )
                partition_keys = list(range(len(keys)))
            else:
                out_columns = columns
                partition_keys = reuse_partition

            if len(needs_conversion) > 0:
                partition_keys = []

            return out_columns, partition_keys, bool(reuse_partition)

        left_columns, left_partition_keys, left_reused = _partition_table(
            self._left,
            self._left_columns,
            self._left_key_indices,
            self._left_needs_conversion,
        )

        right_columns, right_partition_keys, right_reused = _partition_table(
            self._right,
            self._right_columns,
            self._right_key_indices,
            self._right_needs_conversion,
        )

        if left_partition_keys != right_partition_keys:
            if left_reused:
                left_columns = self._hash_partition(
                    self._left_columns,
                    self._left_key_indices,
                    self._left_needs_conversion,
                )
                left_partition_keys = list(range(len(self._left_key_indices)))
            else:
                assert right_reused
                right_columns = self._hash_partition(
                    self._right_columns,
                    self._right_key_indices,
                    self._right_needs_conversion,
                )
                right_partition_keys = list(
                    range(len(self._right_key_indices))
                )

        partition_keys = [
            self._left_key_indices[idx] for idx in left_partition_keys
        ]

        return (left_columns, right_columns, partition_keys)

    def _allocate_output_columns(self, result_storage):
        out_of_range = self._how != JoinTypeCode.INNER

        # We need to allocate only the columns that will appear in the output
        if self._output_common_columns_to_left:
            to_delete = util.snd_set(self._common_indices)

            def left_filter(i):
                return True

            def right_filter(i):
                return i not in to_delete

        else:
            to_delete = util.fst_set(self._common_indices)

            def left_filter(i):
                return i not in to_delete

            def right_filter(i):
                return True

        left_cols_to_copy = util.ifilter(left_filter, self._left_columns)
        right_cols_to_copy = util.ifilter(right_filter, self._right_columns)

        left_on = set(self._left_key_indices)
        left_len = len(self._left_columns)
        left_index_cols = util.ifilter(
            left_filter, [i in left_on for i in range(left_len)]
        )

        right_on = set(self._right_key_indices)
        right_len = len(self._right_columns)
        right_index_cols = util.ifilter(
            right_filter, [i in right_on for i in range(right_len)]
        )

        def _allocate(inputs, index_cols):
            outputs = []
            for input, is_index in zip(inputs, index_cols):
                # For category columns that are used as join keys,
                # we need to allocate the categories columns.
                if is_index and ty.is_categorical_dtype(input.dtype):
                    categories_storage = self._runtime.create_output_storage()
                    categories = categories_storage.create_similar_column(
                        input.children[1].as_column()
                    )
                    result_dtype = ty.CategoricalDtype(
                        categories.as_string_column(), input.dtype.ordered
                    )
                    output = result_storage.create_column(result_dtype, False)
                    output.children.append(
                        result_storage.create_column(ty.uint32, False)
                    )
                    output.children.append(categories)
                    output = output.as_category_column()
                else:
                    output = result_storage.create_similar_column(input, False)

                outputs.append(output)

            for input, output in zip(inputs, outputs):
                if out_of_range or input.nullable:
                    output.set_bitmask(result_storage.create_bitmask())
            return outputs

        left_outputs = _allocate(left_cols_to_copy, left_index_cols)
        right_outputs = _allocate(right_cols_to_copy, right_index_cols)

        return left_outputs, right_outputs

    def _launch_merge_task(self, result_storage, left_outputs, right_outputs):

        right_proj = 0 if self._method == JoinVariantCode.HASH else None

        # Launch the merge task
        plan_merge = Map(self._runtime, OpCode.MERGE)

        plan_merge.add_scalar_arg(self._how, ty.int32)
        plan_merge.add_scalar_arg(self._output_common_columns_to_left, ty.bool)

        plan_merge.add_scalar_arg(len(self._left_key_indices), ty.uint32)
        for idx in self._left_key_indices:
            plan_merge.add_scalar_arg(idx, ty.int32)

        plan_merge.add_scalar_arg(len(self._right_key_indices), ty.uint32)
        for idx in self._right_key_indices:
            plan_merge.add_scalar_arg(idx, ty.int32)

        plan_merge.add_scalar_arg(len(self._common_indices), ty.uint32)
        for lidx, ridx in self._common_indices:
            plan_merge.add_scalar_arg(lidx, ty.int32)
            plan_merge.add_scalar_arg(ridx, ty.int32)

        plan_merge.add_scalar_arg(len(self._left_columns), ty.uint32)
        for value in self._left_columns:
            value.add_to_plan(plan_merge, True)

        plan_merge.add_scalar_arg(len(self._right_columns), ty.uint32)
        for value in self._right_columns:
            value.add_to_plan(plan_merge, True, proj=right_proj)

        plan_merge.add_scalar_arg(len(left_outputs), ty.uint32)
        for output in left_outputs:
            output.add_to_plan_output_only(plan_merge)

        plan_merge.add_scalar_arg(len(right_outputs), ty.uint32)
        for output in right_outputs:
            output.add_to_plan_output_only(plan_merge)

        counts = plan_merge.execute(self._left_columns[0].launch_domain)
        total_count = counts.cast(ty.int64).sum()

        result_storage = plan_merge.promote_output_storage(result_storage)
        self._runtime.register_external_weighted_partition(
            result_storage.default_ipart, counts
        )
        del plan_merge

        # For key columns with categorical dtypes, we need to convert
        # the categories columns into replicated columns
        def _finalize_categories(outputs):
            for output in outputs:
                if ty.is_categorical_dtype(output.dtype):
                    c = output.children[1]
                    if not c.partitioned:
                        continue
                    output.children[1] = c.as_replicated_column()

        _finalize_categories(left_outputs)
        _finalize_categories(right_outputs)

        return result_storage, total_count

    def _finalize_output_columns(self, left_outputs, right_outputs):
        # Generate column names for the output table, and make adjustments
        # to the output columns if necessary.

        # This is a special case where the gathered left index gets the name
        # of the column in the RHS table. We move the output column to the
        # right output list so that everything flows smoothly.
        if self._output_common_columns_to_left and (
            self._left_index and not self._right_index
        ):
            # It's important that we use the number of columns in the LHS,
            # as self._left_columns contains both the value and index columns.
            num_left_value_columns = self._left.num_columns()
            common_columns = left_outputs[num_left_value_columns:]
            left_outputs = left_outputs[:num_left_value_columns]

            adjusted = []
            out_idx, common_idx = 0, 0
            right_common_indices = util.snd_set(self._common_indices)

            for i, _ in enumerate(self._right_columns):
                if i in right_common_indices:
                    adjusted.append(common_columns[common_idx])
                    common_idx += 1
                else:
                    adjusted.append(right_outputs[out_idx])
                    out_idx += 1
            right_outputs = adjusted

        if self._left_index and not self._right_index:
            to_delete = util.fst_set(self._common_indices)

            def left_filter(i):
                return i not in to_delete

            def right_filter(i):
                return True

        else:
            to_delete = util.snd_set(self._common_indices)

            def left_filter(i):
                return True

            def right_filter(i):
                return i not in to_delete

        # Append suffixes to the duplicate column names
        left_column_names = util.ifilter(left_filter, self._left_column_names)
        right_column_names = util.ifilter(
            right_filter, self._right_column_names
        )

        l_suffix, r_suffix = self._suffixes
        common = set(left_column_names) & set(right_column_names)
        left_column_names = [
            (str(c) + l_suffix if c in common else c)
            for c in left_column_names
        ]
        right_column_names = [
            (str(c) + r_suffix if c in common else c)
            for c in right_column_names
        ]
        return (
            left_outputs,
            left_column_names,
            right_outputs,
            right_column_names,
        )

    def _construct_merge_output(self):
        result_storage = self._runtime.create_output_storage()

        (
            left_outputs,
            right_outputs,
        ) = self._allocate_output_columns(result_storage)

        (result_storage, total_count,) = self._launch_merge_task(
            result_storage, left_outputs, right_outputs
        )

        (
            left_outputs,
            left_column_names,
            right_outputs,
            right_column_names,
        ) = self._finalize_output_columns(left_outputs, right_outputs)

        # Filter the index columns and create an index for the output table
        if not self._left_index and not self._right_index:
            result_index = create_range_index(result_storage, total_count)

        else:
            if self._left_index and not self._right_index:
                index_columns = right_outputs[len(right_column_names) :]
                names = util.to_list_if_not_none(self._right._index.name)
            elif not self._left_index and self._right_index:
                index_columns = left_outputs[len(left_column_names) :]
                names = util.to_list_if_not_none(self._left._index.name)
            else:
                index_columns = left_outputs[len(left_column_names) :]
                left_names = util.to_list_if_not_none(self._left._index.name)
                right_names = util.to_list_if_not_none(self._right._index.name)
                names = left_names if left_names == right_names else None

            result_index = create_index_from_columns(
                index_columns,
                total_count,
                names=names,
            )

        # Finally, filter the value columns for the output table
        left_outputs = left_outputs[: len(left_column_names)]
        right_outputs = right_outputs[: len(right_column_names)]

        result_column_names = left_column_names + right_column_names
        result_columns = left_outputs + right_outputs

        return (
            result_index,
            result_column_names,
            result_columns,
            self._partition_keys,
        )

    def perform_merge(self):
        (
            self._left_key_indices,
            self._right_key_indices,
            self._common_indices,
        ) = self._find_join_keys()

        (
            self._left_columns,
            self._right_columns,
        ) = self._prepare_columns()

        self._method = self._parse_merge_method()

        self._unify_key_dtypes()

        if self._method == JoinVariantCode.HASH:
            (
                self._left_columns,
                self._right_columns,
                self._partition_keys,
            ) = self._radix_partition()

        elif self._method == JoinVariantCode.BROADCAST:
            self._right_columns = [
                column.repartition(1) for column in self._right_columns
            ]

        return self._construct_merge_output()

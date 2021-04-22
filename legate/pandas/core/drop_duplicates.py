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

from legate.core import Rect

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode

from .pattern import Map


def _drop_duplicates_one_step(runtime, inputs, subset, keep, radix=1):
    storage = runtime.create_output_storage()

    outputs = [storage.create_similar_column(column) for column in inputs]

    num_pieces = (inputs[0].num_pieces + radix - 1) // radix
    launch_domain = Rect([num_pieces])

    plan = Map(runtime, OpCode.DROP_DUPLICATES_TREE)
    plan.add_scalar_arg(keep.value, ty.int32)
    plan.add_scalar_arg(len(subset), ty.uint32)
    for idx in subset:
        plan.add_scalar_arg(idx, ty.int32)

    plan.add_scalar_arg(radix, ty.uint32)
    for r in range(radix):
        plan.add_scalar_arg(len(inputs), ty.uint32)
        proj_id = runtime.get_radix_functor_id(radix, r)
        for input in inputs:
            input.add_to_plan(plan, True, proj=proj_id)

    plan.add_scalar_arg(len(outputs), ty.uint32)
    for output in outputs:
        output.add_to_plan_output_only(plan)
    counts = plan.execute(launch_domain)

    storage = plan.promote_output_storage(storage)
    return (outputs, storage, counts, outputs[0].num_pieces)


def _drop_duplicates_tree(runtime, inputs, subset, keep):
    (outputs, storage, counts, num_pieces) = _drop_duplicates_one_step(
        runtime, inputs, subset, keep, radix=1
    )
    radix = runtime.radix
    while num_pieces > 1:
        inputs = outputs
        (outputs, storage, counts, num_pieces) = _drop_duplicates_one_step(
            runtime, inputs, subset, keep, radix=radix
        )

    num_pieces = runtime.num_pieces
    outputs = [column.repartition(num_pieces) for column in outputs]
    storage = outputs[0].storage
    volume = counts.cast(ty.int64).sum()

    return (outputs, storage, volume)


def _drop_duplicates_nccl(runtime, inputs, subset, keep):
    storage = runtime.create_output_storage()

    outputs = [storage.create_similar_column(column) for column in inputs]

    plan = Map(runtime, OpCode.DROP_DUPLICATES_NCCL)
    plan.add_scalar_arg(keep.value, ty.int32)
    plan.add_scalar_arg(len(subset), ty.uint32)
    for idx in subset:
        plan.add_scalar_arg(idx, ty.int32)
    plan.add_scalar_arg(len(inputs), ty.uint32)
    for input in inputs:
        input.add_to_plan(plan, True)
    for output in outputs:
        output.add_to_plan_output_only(plan)
    plan.add_future_map(runtime._nccl_comm)
    counts = plan.execute(inputs[0].launch_domain)

    storage = plan.promote_output_storage(storage)
    runtime.register_external_weighted_partition(storage.default_ipart, counts)

    return (outputs, storage, counts.cast(ty.int64).sum())


def drop_duplicates(runtime, inputs, subset, keep):
    keep = runtime.get_keep_method(keep)

    if runtime.use_nccl:
        return _drop_duplicates_nccl(runtime, inputs, subset, keep)
    else:
        return _drop_duplicates_tree(runtime, inputs, subset, keep)

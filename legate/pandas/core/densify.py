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

from legate.pandas.common import types as ty
from legate.pandas.config import OpCode

from .pattern import Map


def to_dense_columns(runtime, inputs):
    storage = runtime.create_output_storage()
    outputs = storage.create_similar_columns(inputs)

    plan = Map(runtime, OpCode.DENSIFY)
    plan.add_scalar_arg(len(inputs), ty.uint32)
    for input in inputs:
        input.add_to_plan(plan, True)
    plan.add_scalar_arg(len(outputs), ty.uint32)
    for output in outputs:
        output.add_to_plan_output_only(plan)

    counts = plan.execute(inputs[0].launch_domain)

    storage = plan.promote_output_storage(storage)
    runtime.register_external_weighted_partition(
        storage.default_ipart, counts.cast(ty.int64)
    )

    return outputs

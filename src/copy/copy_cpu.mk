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

GEN_SRC += copy/concatenate.cc            \
					 copy/copy_if_else.cc           \
					 copy/fill.cc                   \
					 copy/gather.cc                 \
					 copy/materialize.cc            \
					 copy/tasks/compact.cc          \
					 copy/tasks/concatenate.cc      \
					 copy/tasks/copy_if_else.cc     \
					 copy/tasks/densify.cc          \
					 copy/tasks/dropna.cc           \
					 copy/tasks/drop_duplicates.cc  \
					 copy/tasks/fill.cc             \
					 copy/tasks/materialize.cc      \
					 copy/tasks/read_at.cc          \
					 copy/tasks/scatter_by_mask.cc  \
					 copy/tasks/scatter_by_slice.cc \
					 copy/tasks/slice_by_range.cc   \
					 copy/tasks/write_at.cc

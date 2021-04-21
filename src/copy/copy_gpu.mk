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

GEN_SRC += copy/tasks/compact_gpu.cc          \
					 copy/tasks/concatenate_gpu.cc      \
					 copy/tasks/copy_if_else_gpu.cc     \
					 copy/tasks/drop_duplicates_gpu.cc  \
					 copy/tasks/dropna_gpu.cc           \
					 copy/tasks/materialize_gpu.cc      \
					 copy/tasks/read_at_gpu.cc          \
					 copy/tasks/scatter_by_mask_gpu.cc  \
					 copy/tasks/scatter_by_slice_gpu.cc \
					 copy/tasks/slice_by_range_gpu.cc   \

GEN_GPU_SRC += copy/materialize.cu                \
							 copy/tasks/fill.cu                 \
							 copy/tasks/write_at.cu

ifeq ($(strip $(USE_NCCL)),1)
GEN_GPU_SRC += copy/tasks/drop_duplicates_nccl.cu
endif

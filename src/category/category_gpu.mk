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

GEN_SRC += category/conversion.cc                \
					 category/tasks/drop_duplicates_gpu.cc \
					 category/tasks/encode_gpu.cc

GEN_GPU_SRC += category/encode.cu

ifeq ($(strip $(USE_NCCL)),1)
GEN_GPU_SRC += category/drop_duplicates.cu   \
							 category/tasks/encode_nccl.cu
endif

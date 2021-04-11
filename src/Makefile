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

# Flags for directing the runtime makefile what to include
DEBUG    ?= 1		# Include debugging symbols
GPU_ARCH ?= volta
LIBNAME   = liblgpandas

USE_NCCL ?= 0
USE_HIJACK ?= 1
PANDAS_ALLOW_CUDA ?= 1

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif

ifndef ARROW_PATH
$(error ARROW_PATH variable is not defined, aborting build)
endif

ifndef THRUST_PATH
$(error THRUST_PATH variable is not defined, aborting build)
endif

ifeq ($(strip $(USE_CUDA)),1)

ifndef CUDF_PATH
$(error CUDF_PATH variable is not defined, aborting build)
endif

ifndef RMM_PATH
$(error RMM_PATH variable is not defined, aborting build)
endif

ifeq ($(strip $(USE_NCCL)),1)

ifndef NCCL_PATH
$(error NCCL_PATH variable is not defined, aborting build)
endif

endif # ifeq ($(strip $(USE_NCCL)),1)

endif # ifeq ($(strip $(USE_CUDA)),1)

include $(LEGATE_DIR)/share/legate/config.mk

CC_FLAGS ?=
CC_FLAGS += -I. -I$(ARROW_PATH)/include -I$(THRUST_PATH)/

NVCC_FLAGS ?=
NVCC_FLAGS += -I. -I$(CUDF_PATH)/include -I$(CUDF_PATH)/include/libcudf/libcudacxx/ -I$(RMM_PATH)/include -I$(THRUST_PATH)/

ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS += -I. -I$(CUDF_PATH)/include -I$(CUDF_PATH)/include/libcudf/libcudacxx/ -I$(RMM_PATH)/include
endif


LD_FLAGS ?=
LD_FLAGS += -L$(ARROW_PATH)/lib -larrow -lparquet -Wl,-rpath=$(ARROW_PATH)/lib

GEN_SRC		=
GEN_GPU_SRC	=

ifeq ($(strip $(USE_CUDA)),1)
LD_FLAGS += -L$(CUDF_PATH)/lib -lcudf -Wl,-rpath=$(CUDF_PATH)/lib
endif

ifeq ($(strip $(USE_NCCL)),1)
NVCC_FLAGS += -I$(NCCL_PATH)/include -DUSE_NCCL
LD_FLAGS += -L$(NCCL_PATH)/lib -lnccl -Wl,-rpath=$(NCCL_PATH)/lib
endif

ifeq ($(strip $(USE_HIJACK)),0)
NVCC_FLAGS += -DNO_HIJACK
endif

ifeq ($(strip $(USE_CUDA)),0)
CC_FLAGS += -DPANDAS_NO_CUDA -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
USE_CUDA ?= 0
endif

ifeq ($(strip $(PANDAS_DYNAMIC_CUDA_ARCH)),1)
NVCC_FLAGS += -DPANDAS_DYNAMIC_CUDA_ARCH
endif

ifeq ($(strip $(DEBUG)),1)
CC_FLAGS += -DDEBUG_PANDAS
NVCC_FLAGS += -DDEBUG_PANDAS
endif

include cpu_code.mk
include pandas.mk

ifeq ($(strip $(USE_CUDA)),1)
include gpu_code.mk
endif

ifeq ($(strip $(USE_NCCL)),1)
include nccl/nccl.mk
endif

include $(LEGATE_DIR)/share/legate/legate.mk

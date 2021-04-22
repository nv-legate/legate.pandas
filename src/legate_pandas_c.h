/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef __LEGATE_PANDAS_C_H__
#define __LEGATE_PANDAS_C_H__

#include "legate_preamble.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum legate_pandas_type_code_t {
  BOOL_PT   = BOOL_LT,
  INT8_PT   = INT8_LT,
  INT16_PT  = INT16_LT,
  INT32_PT  = INT32_LT,
  INT64_PT  = INT64_LT,
  UINT8_PT  = UINT8_LT,
  UINT16_PT = UINT16_LT,
  UINT32_PT = UINT32_LT,
  UINT64_PT = UINT64_LT,
  FLOAT_PT  = FLOAT_LT,
  DOUBLE_PT = DOUBLE_LT,
  TS_NS_PT,
  RANGE_PT,
  STRING_PT,
  CAT32_PT,
} legate_pandas_type_code_t;

typedef enum legate_pandas_op_code_t {
  ASTYPE = 0,
  BINARY_OP,
  BROADCAST_BINARY_OP,
  BROADCAST_FILLNA,
  BUILD_HISTOGRAM,
  CLEAR_BITMASK,
  COMPACT,
  COMPUTE_RANGE_START,
  COMPUTE_RANGE_STOP,
  COMPUTE_RANGE_VOLUME,
  COMPUTE_SUBRANGE_SIZES,
  CONCATENATE,
  CONTAINS,
  COPY_IF_ELSE,
  COUNT_NULLS,
  CREATE_DIR,
  DENSIFY,
  DROPNA,
  DROP_DUPLICATES,
  ENCODE,
  ENCODE_CATEGORY,
  ENCODE_NCCL,
  EQUALS,
  EVAL_UDF,
  EXPORT_OFFSETS,
  EXTRACT_FIELD,
  FILL,
  FILLNA,
  FINALIZE_NCCL,
  FIND_BOUNDS,
  FIND_BOUNDS_IN_RANGE,
  GLOBAL_PARTITION,
  GROUPBY_REDUCE,
  IMPORT_OFFSETS,
  INIT_BITMASK,
  INIT_NCCL,
  INIT_NCCL_ID,
  ISNA,
  LIBCUDF_INIT,
  LIFT_TO_DOMAIN,
  LOAD_PTX,
  LOCAL_HIST,
  LOCAL_PARTITION,
  MATERIALIZE,
  MERGE,
  NOTNA,
  OFFSETS_TO_RANGES,
  PAD,
  RANGES_TO_OFFSETS,
  READ_AT,
  READ_CSV,
  READ_PARQUET,
  SAMPLE_KEYS,
  SCALAR_BINARY_OP,
  SCALAR_REDUCTION,
  SCAN,
  SCATTER_BY_MASK,
  SCATTER_BY_SLICE,
  SIZES_EQUAL,
  SLICE_BY_RANGE,
  SORT_VALUES,
  SORT_VALUES_NCCL,
  STRING_UOP,
  STRIP,
  TO_BITMASK,
  TO_BOOLMASK,
  TO_BOUNDS,
  TO_COLUMN,
  TO_CSV,
  TO_DATETIME,
  TO_PARQUET,
  UNARY_OP,
  UNARY_REDUCTION,
  WRITE_AT,
  ZFILL,
} legate_pandas_op_code_t;

typedef enum legate_pandas_reduction_op_t {
  PANDAS_REDOP_RANGE_UNION = 0,
} legate_pandas_reduction_op_t;

void legate_pandas_perform_registration();

unsigned legate_pandas_get_cuda_arch();

bool legate_pandas_use_nccl();

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_PANDAS_C_H__

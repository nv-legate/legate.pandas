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

#pragma once

#include <deque>

#include "legate.h"
#include "legate_pandas_c.h"

#ifdef PANDAS_NO_CUDA
#undef LEGATE_USE_CUDA
#endif

namespace legate {
namespace pandas {

using OpCode = legate_pandas_op_code_t;

enum class TypeCode : int {
  BOOL   = BOOL_PT,
  INT8   = INT8_PT,
  INT16  = INT16_PT,
  INT32  = INT32_PT,
  INT64  = INT64_PT,
  UINT8  = UINT8_PT,
  UINT16 = UINT16_PT,
  UINT32 = UINT32_PT,
  UINT64 = UINT64_PT,
  FLOAT  = FLOAT_PT,
  DOUBLE = DOUBLE_PT,
  TS_NS  = TS_NS_PT,
  RANGE  = RANGE_PT,
  STRING = STRING_PT,
  CAT32  = CAT32_PT,
  INVALID,
};

template <typename>
struct PandasTypeCodeOf {
  static constexpr TypeCode value = TypeCode::INVALID;
};
template <>
struct PandasTypeCodeOf<float> {
  static constexpr TypeCode value = TypeCode::FLOAT;
};
template <>
struct PandasTypeCodeOf<double> {
  static constexpr TypeCode value = TypeCode::DOUBLE;
};
template <>
struct PandasTypeCodeOf<int8_t> {
  static constexpr TypeCode value = TypeCode::INT8;
};
template <>
struct PandasTypeCodeOf<int16_t> {
  static constexpr TypeCode value = TypeCode::INT16;
};
template <>
struct PandasTypeCodeOf<int32_t> {
  static constexpr TypeCode value = TypeCode::INT32;
};
template <>
struct PandasTypeCodeOf<int64_t> {
  static constexpr TypeCode value = TypeCode::INT64;
};
template <>
struct PandasTypeCodeOf<uint8_t> {
  static constexpr TypeCode value = TypeCode::UINT8;
};
template <>
struct PandasTypeCodeOf<uint16_t> {
  static constexpr TypeCode value = TypeCode::UINT16;
};
template <>
struct PandasTypeCodeOf<uint32_t> {
  static constexpr TypeCode value = TypeCode::UINT32;
};
template <>
struct PandasTypeCodeOf<uint64_t> {
  static constexpr TypeCode value = TypeCode::UINT64;
};
template <>
struct PandasTypeCodeOf<bool> {
  static constexpr TypeCode value = TypeCode::BOOL;
};
template <>
struct PandasTypeCodeOf<std::string> {
  static constexpr TypeCode value = TypeCode::STRING;
};
template <>
struct PandasTypeCodeOf<Legion::Rect<1>> {
  static constexpr TypeCode value = TypeCode::RANGE;
};

template <typename T>
constexpr TypeCode pandas_type_code_of = PandasTypeCodeOf<T>::value;

template <TypeCode CODE>
struct PandasTypeOf {
  using type = void;
};
template <>
struct PandasTypeOf<TypeCode::INT8> {
  using type = int8_t;
};
template <>
struct PandasTypeOf<TypeCode::INT16> {
  using type = int16_t;
};
template <>
struct PandasTypeOf<TypeCode::INT32> {
  using type = int32_t;
};
template <>
struct PandasTypeOf<TypeCode::INT64> {
  using type = int64_t;
};
template <>
struct PandasTypeOf<TypeCode::TS_NS> {
  using type = int64_t;
};
template <>
struct PandasTypeOf<TypeCode::UINT8> {
  using type = uint8_t;
};
template <>
struct PandasTypeOf<TypeCode::UINT16> {
  using type = uint16_t;
};
template <>
struct PandasTypeOf<TypeCode::UINT32> {
  using type = uint32_t;
};
template <>
struct PandasTypeOf<TypeCode::UINT64> {
  using type = uint64_t;
};
template <>
struct PandasTypeOf<TypeCode::FLOAT> {
  using type = float;
};
template <>
struct PandasTypeOf<TypeCode::DOUBLE> {
  using type = double;
};
template <>
struct PandasTypeOf<TypeCode::BOOL> {
  using type = bool;
};
template <>
struct PandasTypeOf<TypeCode::RANGE> {
  using type = Legion::Rect<1>;
};
template <>
struct PandasTypeOf<TypeCode::STRING> {
  using type = std::string;
};

template <TypeCode CODE>
using pandas_type_of = typename PandasTypeOf<CODE>::type;

template <TypeCode CODE>
struct is_primitive_type : std::true_type {
};
template <>
struct is_primitive_type<TypeCode::STRING> : std::false_type {
};
template <>
struct is_primitive_type<TypeCode::CAT32> : std::false_type {
};

template <TypeCode CODE>
struct is_numeric_type : std::true_type {
};
template <>
struct is_numeric_type<TypeCode::TS_NS> : std::false_type {
};
template <>
struct is_numeric_type<TypeCode::STRING> : std::false_type {
};
template <>
struct is_numeric_type<TypeCode::CAT32> : std::false_type {
};

template <TypeCode CODE>
struct is_integral_type : is_numeric_type<CODE> {
};
template <>
struct is_integral_type<TypeCode::FLOAT> : std::false_type {
};
template <>
struct is_integral_type<TypeCode::DOUBLE> : std::false_type {
};

TypeCode to_storage_type_code(TypeCode code);

bool is_primitive_type_code(TypeCode code);

size_t size_of_type(TypeCode code);

enum class UnaryOpCode : int {
  ABS        = 0,
  BIT_INVERT = 1,
};

enum class BinaryOpCode : int {
  ADD           = 0,
  SUB           = 1,
  MUL           = 2,
  DIV           = 3,
  FLOOR_DIV     = 4,
  MOD           = 5,
  POW           = 6,
  EQUAL         = 7,
  NOT_EQUAL     = 8,
  LESS          = 9,
  GREATER       = 10,
  LESS_EQUAL    = 11,
  GREATER_EQUAL = 12,
  BITWISE_AND   = 13,
  BITWISE_OR    = 14,
  BITWISE_XOR   = 15,
};

enum class ProjectionCode : int {
  PROJ_RADIX_4_0 = 0,
  PROJ_RADIX_4_1 = 1,
  PROJ_RADIX_4_2 = 2,
  PROJ_RADIX_4_3 = 3,
  LAST_PROJ      = 4,
};

enum class ShardingCode : int {
  SHARD_TILE = 0,
  LAST_SHARD = 1,
};

enum class KeepMethod : int {
  FIRST = 0,
  LAST  = 1,
  NONE  = 2,
};

enum class JoinTypeCode : int {
  INNER = 0,
  LEFT  = 1,
  RIGHT = 2,
  OUTER = 3,
};

enum class AggregationCode : int {
  SUM   = 0,
  MIN   = 1,
  MAX   = 2,
  COUNT = 3,
  PROD  = 4,
  MEAN  = 5,
  VAR   = 6,
  STD   = 7,
  SIZE  = 8,
  ANY   = 9,
  ALL   = 10,
  SQSUM = 11,  // Used internally in the CPU implementation and invisible to the user
};

enum class DatetimeFieldCode : int {
  YEAR    = 0,
  MONTH   = 1,
  DAY     = 2,
  HOUR    = 3,
  MINUTE  = 4,
  SECOND  = 5,
  WEEKDAY = 6,
};

enum class StringMethods : int {
  LOWER    = 0,
  UPPER    = 1,
  SWAPCASE = 2,
};

enum class PadSideCode : int {
  LEFT  = 0,
  RIGHT = 1,
  BOTH  = 2,
};

enum class CompressionType : int {
  UNCOMPRESSED = 0,
  SNAPPY       = 1,
  GZIP         = 2,
  BROTLI       = 3,
  BZ2          = 4,
  ZIP          = 5,
  XZ           = 6,
};

enum class PartitionId : Legion::Color {
  EQUAL = 1,
};

class LegatePandas {
 public:
  // Record variants for all our tasks
  static void record_variant(Legion::TaskID tid,
                             const char *task_name,
                             const Legion::CodeDescriptor &desc,
                             Legion::ExecutionConstraintSet &execution_constraints,
                             Legion::TaskLayoutConstraintSet &layout_constraints,
                             Legion::VariantID var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size);
  // Runtime registration callback
  static void registration_callback(Legion::Machine m,
                                    Legion::Runtime *rt,
                                    const std::set<Legion::Processor> &local_procs);

 public:
  struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
   public:
    PendingTaskVariant(void)
      : Legion::TaskVariantRegistrar(), task_name(NULL), var(LEGATE_NO_VARIANT)
    {
    }
    PendingTaskVariant(Legion::TaskID tid,
                       bool global,
                       const char *var_name,
                       const char *t_name,
                       const Legion::CodeDescriptor &desc,
                       Legion::VariantID v,
                       size_t ret)
      : Legion::TaskVariantRegistrar(tid, global, var_name),
        task_name(t_name),
        descriptor(desc),
        var(v),
        ret_size(ret)
    {
    }

   public:
    const char *task_name;
    Legion::CodeDescriptor descriptor;
    Legion::VariantID var;
    size_t ret_size;
  };
  static std::deque<PendingTaskVariant> &get_pending_task_variants(void);
};

template <typename T>
class PandasTask : public LegateTask<T> {
 public:
  // Almost all Pandas tasks are variadic, so here we pretend that they
  // take no region at all and handle layout constraints differently.
  static const int REGIONS = 0;

  // Record variants for all our tasks
  template <typename VARIANT>
  static void record_variant(Legion::TaskID tid,
                             const Legion::CodeDescriptor &desc,
                             Legion::ExecutionConstraintSet &execution_constraints,
                             Legion::TaskLayoutConstraintSet &layout_constraints,
                             VARIANT var,
                             Legion::Processor::Kind kind,
                             bool leaf,
                             bool inner,
                             bool idempotent,
                             size_t ret_size)
  {
    // For this just turn around and call this on the base LegatePandas
    // type so it will deduplicate across all task kinds
    LegatePandas::record_variant(tid,
                                 T::task_name(),
                                 desc,
                                 execution_constraints,
                                 layout_constraints,
                                 var,
                                 kind,
                                 leaf,
                                 inner,
                                 idempotent,
                                 ret_size);
  }
};

#define NUM_PROJ (static_cast<int>(ProjectionCode::LAST_PROJ))

#define NUM_SHARD (static_cast<int>(ShardingCode::LAST_SHARD))

}  // namespace pandas
}  // namespace legate

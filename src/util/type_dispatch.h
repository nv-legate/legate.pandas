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

#include "pandas.h"

namespace legate {
namespace pandas {

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(TypeCode code, Functor f, Fnargs &&... args)
{
  switch (code) {
    case TypeCode::BOOL: {
      return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT8: {
      return f.template operator()<TypeCode::INT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT16: {
      return f.template operator()<TypeCode::INT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT32: {
      return f.template operator()<TypeCode::INT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT64: {
      return f.template operator()<TypeCode::INT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT8: {
      return f.template operator()<TypeCode::UINT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT16: {
      return f.template operator()<TypeCode::UINT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT32: {
      return f.template operator()<TypeCode::UINT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT64: {
      return f.template operator()<TypeCode::UINT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::FLOAT: {
      return f.template operator()<TypeCode::FLOAT>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::DOUBLE: {
      return f.template operator()<TypeCode::DOUBLE>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::TS_NS: {
      return f.template operator()<TypeCode::TS_NS>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::STRING: {
      return f.template operator()<TypeCode::STRING>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::CAT32: {
      return f.template operator()<TypeCode::CAT32>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_numeric_only(TypeCode code, Functor f, Fnargs &&... args)
{
  switch (code) {
    case TypeCode::BOOL: {
      return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT8: {
      return f.template operator()<TypeCode::INT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT16: {
      return f.template operator()<TypeCode::INT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT32: {
      return f.template operator()<TypeCode::INT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT64: {
      return f.template operator()<TypeCode::INT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT8: {
      return f.template operator()<TypeCode::UINT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT16: {
      return f.template operator()<TypeCode::UINT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT32: {
      return f.template operator()<TypeCode::UINT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT64: {
      return f.template operator()<TypeCode::UINT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::FLOAT: {
      return f.template operator()<TypeCode::FLOAT>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::DOUBLE: {
      return f.template operator()<TypeCode::DOUBLE>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::TS_NS: {
      return f.template operator()<TypeCode::TS_NS>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_primitive_only(TypeCode code, Functor f, Fnargs &&... args)
{
  switch (code) {
    case TypeCode::BOOL: {
      return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT8: {
      return f.template operator()<TypeCode::INT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT16: {
      return f.template operator()<TypeCode::INT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT32: {
      return f.template operator()<TypeCode::INT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::INT64: {
      return f.template operator()<TypeCode::INT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT8: {
      return f.template operator()<TypeCode::UINT8>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT16: {
      return f.template operator()<TypeCode::UINT16>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT32: {
      return f.template operator()<TypeCode::UINT32>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::UINT64: {
      return f.template operator()<TypeCode::UINT64>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::FLOAT: {
      return f.template operator()<TypeCode::FLOAT>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::DOUBLE: {
      return f.template operator()<TypeCode::DOUBLE>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::TS_NS: {
      return f.template operator()<TypeCode::TS_NS>(std::forward<Fnargs>(args)...);
    }
    case TypeCode::RANGE: {
      return f.template operator()<TypeCode::RANGE>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<TypeCode::BOOL>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(BinaryOpCode code, Functor f, Fnargs &&... args)
{
  switch (code) {
    case BinaryOpCode::ADD: {
      return f.template operator()<BinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::SUB: {
      return f.template operator()<BinaryOpCode::SUB>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::MUL: {
      return f.template operator()<BinaryOpCode::MUL>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::DIV: {
      return f.template operator()<BinaryOpCode::DIV>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::FLOOR_DIV: {
      return f.template operator()<BinaryOpCode::FLOOR_DIV>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::MOD: {
      return f.template operator()<BinaryOpCode::MOD>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::POW: {
      return f.template operator()<BinaryOpCode::POW>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::EQUAL: {
      return f.template operator()<BinaryOpCode::EQUAL>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::NOT_EQUAL: {
      return f.template operator()<BinaryOpCode::NOT_EQUAL>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::LESS: {
      return f.template operator()<BinaryOpCode::LESS>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::GREATER: {
      return f.template operator()<BinaryOpCode::GREATER>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::LESS_EQUAL: {
      return f.template operator()<BinaryOpCode::LESS_EQUAL>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::GREATER_EQUAL: {
      return f.template operator()<BinaryOpCode::GREATER_EQUAL>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::BITWISE_AND: {
      return f.template operator()<BinaryOpCode::BITWISE_AND>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::BITWISE_OR: {
      return f.template operator()<BinaryOpCode::BITWISE_OR>(std::forward<Fnargs>(args)...);
    }
    case BinaryOpCode::BITWISE_XOR: {
      return f.template operator()<BinaryOpCode::BITWISE_XOR>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<BinaryOpCode::ADD>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(AggregationCode code, Functor f, Fnargs &&... args)
{
  switch (code) {
    case AggregationCode::SUM: {
      return f.template operator()<AggregationCode::SUM>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::MIN: {
      return f.template operator()<AggregationCode::MIN>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::MAX: {
      return f.template operator()<AggregationCode::MAX>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::COUNT: {
      return f.template operator()<AggregationCode::COUNT>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::PROD: {
      return f.template operator()<AggregationCode::PROD>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::MEAN: {
      return f.template operator()<AggregationCode::MEAN>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::VAR: {
      return f.template operator()<AggregationCode::VAR>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::STD: {
      return f.template operator()<AggregationCode::STD>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::SIZE: {
      return f.template operator()<AggregationCode::SIZE>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::ANY: {
      return f.template operator()<AggregationCode::ANY>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::ALL: {
      return f.template operator()<AggregationCode::ALL>(std::forward<Fnargs>(args)...);
    }
    case AggregationCode::SQSUM: {
      return f.template operator()<AggregationCode::SQSUM>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<AggregationCode::SUM>(std::forward<Fnargs>(args)...);
}

}  // namespace pandas
}  // namespace legate

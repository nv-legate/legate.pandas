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
namespace reduction {

template <AggregationCode CODE, typename T>
struct Op;

template <typename T>
struct Op<AggregationCode::SUM, T> {
  using result_t = T;

  constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }

  static constexpr T identity(void) { return Legion::SumReduction<T>::identity; }
};

template <typename T>
struct Op<AggregationCode::MIN, T> {
  using result_t = T;

  constexpr T operator()(const T &lhs, const T &rhs) const { return std::min(lhs, rhs); }

  static constexpr T identity(void) { return Legion::MinReduction<T>::identity; }
};

template <>
struct Op<AggregationCode::MIN, std::string> {
  using result_t = std::string;

  std::string operator()(const std::string &lhs, const std::string &rhs) const
  {
    return std::min(lhs, rhs);
  }

  static std::string identity(void)
  {
    assert(false);
    return "";
  }
};

template <typename T>
struct Op<AggregationCode::MAX, T> {
  using result_t = T;

  constexpr T operator()(const T &lhs, const T &rhs) const { return std::max(lhs, rhs); }

  static constexpr T identity(void) { return Legion::MaxReduction<T>::identity; }
};

template <>
struct Op<AggregationCode::MAX, std::string> {
  using result_t = std::string;

  std::string operator()(const std::string &lhs, const std::string &rhs) const
  {
    return std::max(lhs, rhs);
  }

  static std::string identity(void)
  {
    assert(false);
    return "";
  }
};

template <typename T>
struct Op<AggregationCode::PROD, T> {
  using result_t = T;

  constexpr T operator()(const T &lhs, const T &rhs) const { return lhs * rhs; }

  static constexpr T identity(void) { return Legion::ProdReduction<T>::identity; }
};

template <typename T>
struct Op<AggregationCode::ANY, T> {
  using result_t = bool;

  constexpr bool operator()(const bool &lhs, const T &rhs) const
  {
    return lhs || static_cast<bool>(rhs);
  }

  static constexpr T identity(void) { return false; }
};

template <typename T>
struct Op<AggregationCode::ALL, T> {
  using result_t = bool;

  constexpr bool operator()(const bool &lhs, const T &rhs) const
  {
    return lhs && static_cast<bool>(rhs);
  }

  static constexpr bool identity(void) { return true; }
};

template <typename T>
struct Op<AggregationCode::SQSUM, T> {
  using result_t = T;

  constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs * rhs; }

  static constexpr T identity(void) { return Legion::SumReduction<T>::identity; }
};

template <AggregationCode CODE>
struct is_numeric_aggregation : std::false_type {
};

template <>
struct is_numeric_aggregation<AggregationCode::SUM> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::PROD> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::MEAN> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::VAR> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::STD> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::ANY> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::ALL> : std::true_type {
};
template <>
struct is_numeric_aggregation<AggregationCode::SQSUM> : std::true_type {
};
// FIXME: These are not technically numeric aggregation, but the current implementation
//        assumes that the values are numeric. We should change these to generic
//        aggregations, once we add a proper implementation for them.
// template <>
// struct is_numeric_aggregation<AggregationCode::MIN> : std::true_type {
//};
// template <>
// struct is_numeric_aggregation<AggregationCode::MAX> : std::true_type {
//};

template <AggregationCode CODE>
struct is_compound_aggregation : std::false_type {
};

template <>
struct is_compound_aggregation<AggregationCode::MEAN> : std::true_type {
};
template <>
struct is_compound_aggregation<AggregationCode::VAR> : std::true_type {
};
template <>
struct is_compound_aggregation<AggregationCode::STD> : std::true_type {
};

}  // namespace reduction
}  // namespace pandas
}  // namespace legate

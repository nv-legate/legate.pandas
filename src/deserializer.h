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

#include <tuple>

#include "pandas.h"
#include "column/region_arg.h"
#include "column/output_region_arg.h"
#include "scalar/scalar.h"

namespace legate {
namespace pandas {

template <class T>
class FromFuture {
 public:
  FromFuture()                   = default;
  FromFuture(const FromFuture &) = default;

  FromFuture(const T &value) : value_(value) {}

  inline operator T() const { return value(); }

  const T &value() const { return value_; }

 private:
  T value_;
};

struct FromRawFuture {
  FromRawFuture()                      = default;
  FromRawFuture(const FromRawFuture &) = default;

  const void *rawptr_{nullptr};
  size_t size_{0};
};

template <class T, T default_value>
class MaybeFuture {
 public:
  MaybeFuture() : value_(default_value) {}
  MaybeFuture(const MaybeFuture &) = default;

  MaybeFuture(const T &value) : value_(value) {}

  const T &value() const { return value_; }

 private:
  T value_;
};

template <typename T>
class Maybe {
 public:
  Maybe() : has_value_(false), value_() {}
  Maybe(Maybe<T> &&other) : has_value_(other.has_value_), value_(std::forward<T>(other.value_))
  {
    other.has_value_ = false;
  }
  Maybe(T &&value) : has_value_(true), value_(std::forward<T>(value)) {}

  bool valid() const { return has_value_; }
  void operator=(T &&value)
  {
    has_value_ = true;
    value_     = std::forward<T>(value);
  }

  const T &operator*() const
  {
#ifdef DEBUG_PANDAS
    assert(has_value_);
#endif
    return value_;
  }

  T &operator*()
  {
#ifdef DEBUG_PANDAS
    assert(has_value_);
#endif
    return value_;
  }

 private:
  bool has_value_;
  T value_;
};

template <typename T>
struct MiniSpan {
 public:
  MiniSpan(T *data, size_t size) : data_(data), size_(size) {}

 public:
  decltype(auto) operator[](size_t pos)
  {
    assert(pos < size_);
    return data_[pos];
  }

 public:
  decltype(auto) subspan(size_t off)
  {
    assert(off <= size_);
    return MiniSpan(data_ + off, size_ - off);
  }

 private:
  T *data_;
  size_t size_;
};

class Deserializer {
 public:
  Deserializer(const Legion::Task *task, const std::vector<Legion::PhysicalRegion> &regions);

 public:
  friend void deserialize(Deserializer &ctx, __half &value);
  friend void deserialize(Deserializer &ctx, float &value);
  friend void deserialize(Deserializer &ctx, double &value);
  friend void deserialize(Deserializer &ctx, std::uint64_t &value);
  friend void deserialize(Deserializer &ctx, std::uint32_t &value);
  friend void deserialize(Deserializer &ctx, std::uint16_t &value);
  friend void deserialize(Deserializer &ctx, std::uint8_t &value);
  friend void deserialize(Deserializer &ctx, std::int64_t &value);
  friend void deserialize(Deserializer &ctx, std::int32_t &value);
  friend void deserialize(Deserializer &ctx, std::int16_t &value);
  friend void deserialize(Deserializer &ctx, std::int8_t &value);
  friend void deserialize(Deserializer &ctx, std::string &value);
  friend void deserialize(Deserializer &ctx, TypeCode &code);
  friend void deserialize(Deserializer &ctx, UnaryOpCode &code);
  friend void deserialize(Deserializer &ctx, BinaryOpCode &code);
  friend void deserialize(Deserializer &ctx, AggregationCode &code);
  friend void deserialize(Deserializer &ctx, DatetimeFieldCode &code);
  friend void deserialize(Deserializer &ctx, CompressionType &code);
  friend void deserialize(Deserializer &ctx, JoinTypeCode &code);
  friend void deserialize(Deserializer &ctx, bool &value);
  friend void deserialize(Deserializer &ctx, Legion::PhysicalRegion &pr, Legion::FieldID &fid);

 public:
  template <typename T1, typename T2>
  friend void deserialize(Deserializer &ctx, std::pair<T1, T2> &pair);
  template <typename T>
  void deserialize(Deserializer &ctx, std::vector<T> &vector, bool resize = true);

 public:
  template <typename T>
  void deserialize(Deserializer &ctx, Maybe<T> &maybe);

 public:
  template <class T, int N>
  friend Legion::Rect<N> deserialize(Deserializer &ctx, AccessorWO<T, N> &accessor);
  template <class T, int N>
  friend Legion::Rect<N> deserialize(Deserializer &ctx, AccessorRO<T, N> &accessor);
  template <class T, int N>
  friend Legion::Rect<N> deserialize(Deserializer &ctx, AccessorRW<T, N> &accessor);
  template <bool READ, int DIM>
  friend void deserialize(Deserializer &ctx, RegionArg<READ, DIM> &arg);
  friend void deserialize(Deserializer &ctx, OutputRegionArg &arg);
  friend void deserialize(Deserializer &ctx, Scalar &scalar);

 public:
  template <class T>
  friend void deserialize(Deserializer &ctx, FromFuture<T> &scalar);
  template <class T>
  friend void deserialize_from_future(Deserializer &ctx, T &scalar);
  friend void deserialize(Deserializer &ctx, FromRawFuture &scalar);

 private:
  const Legion::Task *task_;
  MiniSpan<const Legion::PhysicalRegion> regions_;
  MiniSpan<const Legion::Future> futures_;
  LegateDeserializer deserializer_;
  std::vector<Legion::OutputRegion> outputs_;
};

template <typename T1, typename T2>
void deserialize(Deserializer &ctx, std::pair<T1, T2> &pair)
{
  deserialize(ctx, pair.first);
  deserialize(ctx, pair.second);
}

template <typename T>
void deserialize(Deserializer &ctx, std::vector<T> &vector, bool resize = true)
{
  if (resize) {
    uint32_t size = 0;
    deserialize(ctx, size);
    vector.resize(size);
  }
  for (auto &elem : vector) deserialize(ctx, elem);
}

template <>
inline void deserialize(Deserializer &ctx, std::vector<bool> &vector, bool resize)
{
  if (resize) {
    uint32_t size = 0;
    deserialize(ctx, size);
    vector.resize(size);
  }
  for (auto it = vector.begin(); it != vector.end(); ++it) {
    bool value;
    deserialize(ctx, value);
    *it = value;
  }
}

template <typename T>
void deserialize(Deserializer &ctx, Maybe<T> &maybe)
{
  bool has_value{false};
  deserialize(ctx, has_value);
  if (has_value) {
    T value;
    deserialize(ctx, value);
    maybe = std::move(value);
  }
}

template <class T, int N>
Legion::Rect<N> deserialize(Deserializer &ctx, AccessorWO<T, N> &accessor)
{
  ctx.deserializer_.unpack_32bit_int();
  uint32_t idx = ctx.deserializer_.unpack_32bit_uint();
  Legion::Rect<N> shape(ctx.regions_[idx]);
  if (shape.volume() > 0)
    accessor = ctx.deserializer_.unpack_accessor_WO<T, N>(ctx.regions_[idx], shape);
  return shape;
}

template <class T, int N>
Legion::Rect<N> deserialize(Deserializer &ctx, AccessorRO<T, N> &accessor)
{
  ctx.deserializer_.unpack_32bit_int();
  uint32_t idx = ctx.deserializer_.unpack_32bit_uint();
  Legion::Rect<N> shape(ctx.regions_[idx]);
  if (shape.volume() > 0)
    accessor = ctx.deserializer_.unpack_accessor_RO<T, N>(ctx.regions_[idx], shape);
  return shape;
}

template <class T, int N>
Legion::Rect<N> deserialize(Deserializer &ctx, AccessorRW<T, N> &accessor)
{
  ctx.deserializer_.unpack_32bit_int();
  uint32_t idx = ctx.deserializer_.unpack_32bit_uint();
  Legion::Rect<N> shape(ctx.regions_[idx]);
  if (shape.volume() > 0)
    accessor = ctx.deserializer_.unpack_accessor_RW<T, N>(ctx.regions_[idx], shape);
  return shape;
}

template <bool READ, int DIM>
void deserialize(Deserializer &ctx, RegionArg<READ, DIM> &arg)
{
  const auto code = static_cast<TypeCode>(ctx.deserializer_.unpack_32bit_int());
  const auto idx  = ctx.deserializer_.unpack_32bit_uint();
  const auto fid  = static_cast<Legion::FieldID>(ctx.deserializer_.unpack_32bit_int());
  const auto M    = ctx.deserializer_.unpack_32bit_int();
#ifdef DEBUG_PANDAS
  assert(M == 0);
#endif
  arg = RegionArg<READ, DIM>{code, ctx.regions_[idx], fid};
}

template <class T>
void deserialize(Deserializer &ctx, FromFuture<T> &scalar)
{
  // grab the scalar out of the first future
  scalar = FromFuture<T>{ctx.futures_[0].get_result<T>()};

  // discard the first future
  ctx.futures_ = ctx.futures_.subspan(1);
}

template <class T>
void deserialize_from_future(Deserializer &ctx, T &scalar)
{
  FromFuture<T> fut;
  deserialize(ctx, fut);
  scalar = fut.value();
}

template <class T, T v>
void deserialize(Deserializer &ctx, MaybeFuture<T, v> &scalar)
{
  bool has_value = false;
  deserialize(ctx, has_value);

  if (has_value) {
    // grab the scalar out of the first future
    scalar = MaybeFuture<T, v>{ctx.futures_[0].get_result<T>()};

    // discard the first future
    ctx.futures_ = ctx.futures_.subspan(1);
  } else {
    scalar = MaybeFuture<T, v>{};
  }
}

}  // namespace pandas
}  // namespace legate

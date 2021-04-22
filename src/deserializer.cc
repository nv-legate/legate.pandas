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

#include "deserializer.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

using namespace Legion;

Deserializer::Deserializer(const Task *task, const std::vector<PhysicalRegion> &regions)
  : task_{task},
    regions_{regions.data(), regions.size()},
    futures_{task->futures.data(), task->futures.size()},
    deserializer_{task->args, task->arglen},
    outputs_()
{
  auto runtime = Runtime::get_runtime();
  auto ctx     = Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

void deserialize(Deserializer &ctx, __half &value) { value = ctx.deserializer_.unpack_half(); }

void deserialize(Deserializer &ctx, float &value) { value = ctx.deserializer_.unpack_float(); }

void deserialize(Deserializer &ctx, double &value) { value = ctx.deserializer_.unpack_double(); }

void deserialize(Deserializer &ctx, std::uint64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_uint();
}

void deserialize(Deserializer &ctx, std::uint32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_uint();
}

void deserialize(Deserializer &ctx, std::uint16_t &value)
{
  value = ctx.deserializer_.unpack_16bit_uint();
}

void deserialize(Deserializer &ctx, std::uint8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_uint();
}

void deserialize(Deserializer &ctx, std::int64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_int();
}

void deserialize(Deserializer &ctx, std::int16_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_int();
}

void deserialize(Deserializer &ctx, std::string &value)
{
  value = ctx.deserializer_.unpack_string();
}

void deserialize(Deserializer &ctx, TypeCode &code)
{
  code = static_cast<TypeCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, UnaryOpCode &code)
{
  code = static_cast<UnaryOpCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, BinaryOpCode &code)
{
  code = static_cast<BinaryOpCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, AggregationCode &code)
{
  code = static_cast<AggregationCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, DatetimeFieldCode &code)
{
  code = static_cast<DatetimeFieldCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, CompressionType &code)
{
  code = static_cast<CompressionType>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, JoinTypeCode &code)
{
  code = static_cast<JoinTypeCode>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, KeepMethod &code)
{
  code = static_cast<KeepMethod>(ctx.deserializer_.unpack_32bit_int());
}

void deserialize(Deserializer &ctx, bool &value) { value = ctx.deserializer_.unpack_bool(); }

void deserialize(Deserializer &ctx, Legion::PhysicalRegion &pr, Legion::FieldID &fid)
{
  uint32_t idx = ctx.deserializer_.unpack_32bit_uint();
  pr           = ctx.regions_[idx];
  fid          = ctx.deserializer_.unpack_32bit_int();
  ctx.deserializer_.unpack_32bit_int();
}

void deserialize(Deserializer &ctx, OutputRegionArg &arg)
{
  const auto code = static_cast<TypeCode>(ctx.deserializer_.unpack_32bit_int());
  const auto idx  = ctx.deserializer_.unpack_32bit_uint();
  const auto fid  = static_cast<Legion::FieldID>(ctx.deserializer_.unpack_32bit_int());
  const auto M    = ctx.deserializer_.unpack_32bit_int();
#ifdef DEBUG_PANDAS
  assert(M == 0);
#endif
  arg = OutputRegionArg(code, ctx.outputs_[idx], fid);
}

void deserialize(Deserializer &ctx, Scalar &arg)
{
  arg          = ctx.futures_[0].get_result<Scalar>();
  ctx.futures_ = ctx.futures_.subspan(1);
}

void deserialize(Deserializer &ctx, FromRawFuture &scalar)
{
  scalar = FromRawFuture{
    ctx.futures_[0].get_buffer(Memory::SYSTEM_MEM),
    ctx.futures_[0].get_untyped_size(),
  };
  assert(nullptr != scalar.rawptr_);

  // discard the first future
  ctx.futures_ = ctx.futures_.subspan(1);
}

}  // namespace pandas
}  // namespace legate

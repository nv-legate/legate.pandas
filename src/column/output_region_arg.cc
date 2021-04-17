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

#include "column/output_region_arg.h"
#include "util/type_dispatch.h"

using namespace Legion;

namespace legate {
namespace pandas {

namespace detail {

struct Destroy {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  void operator()(void *buffer)
  {
    using VAL = pandas_type_of<CODE>;
    delete static_cast<DeferredBuffer<VAL, 1> *>(buffer);
  }
};

struct Allocate {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  void *operator()(Legion::OutputRegion &out, size_t size, size_t alignment, FieldID fid)
  {
    using VAL = pandas_type_of<CODE>;
    Rect<1> bounds(0, size - 1);
    void *buffer = new DeferredBuffer<VAL, 1>(bounds, out.target_memory(), NULL, alignment);
    out.return_data(fid, *static_cast<DeferredBuffer<VAL, 1> *>(buffer));
    return buffer;
  }

  template <TypeCode CODE, std::enable_if_t<!is_primitive_type<CODE>::value> * = nullptr>
  void *operator()(Legion::OutputRegion &out, size_t size, size_t alignment, FieldID fid)
  {
    out.return_data(size, fid, nullptr);
    return nullptr;
  }
};

struct UntypedPtr {
  template <TypeCode CODE, std::enable_if_t<is_primitive_type<CODE>::value> * = nullptr>
  void *operator()(void *buffer)
  {
    using VAL = pandas_type_of<CODE>;
    return static_cast<DeferredBuffer<VAL, 1> *>(buffer)->ptr(0);
  }
};

}  // namespace detail

OutputRegionArg::OutputRegionArg() : code(TypeCode::INVALID), out_(), fid_(-1U), buffer_(nullptr) {}

OutputRegionArg::OutputRegionArg(TypeCode code,
                                 const Legion::OutputRegion &out,
                                 Legion::FieldID fid)
  : code(code), out_(out), fid_(fid), buffer_(nullptr)
{
}

OutputRegionArg::~OutputRegionArg()
{
  if (nullptr != buffer_) type_dispatch_primitive_only(code, detail::Destroy{}, buffer_);
}

OutputRegionArg::OutputRegionArg(OutputRegionArg &&other)
  : code(other.code), out_(other.out_), fid_(other.fid_), buffer_(other.buffer_)
{
  other.buffer_ = nullptr;
}

OutputRegionArg &OutputRegionArg::operator=(OutputRegionArg &&other)
{
  code          = other.code;
  out_          = other.out_;
  fid_          = other.fid_;
  buffer_       = other.buffer_;
  other.buffer_ = nullptr;
  return *this;
}

void OutputRegionArg::allocate(size_t num_elements, size_t alignment)
{
#ifdef DEBUG_PANDAS
  assert(nullptr == buffer_);
#endif

  if (num_elements == 0) {
    out_.return_data(0, fid_, nullptr);
    return;
  }

  buffer_ = type_dispatch(code, detail::Allocate{}, out_, num_elements, alignment, fid_);
}

void OutputRegionArg::return_from_instance(Realm::RegionInstance instance,
                                           size_t num_elements,
                                           size_t elem_size)
{
  out_.return_data(fid_, instance, elem_size, &num_elements);
}

Rect<1> OutputRegionArg::shape() const
{
#ifdef DEBUG_PANDAS
  assert(out_.is_valid_output_region());
#endif
  IndexSpace is = out_.get_logical_region().get_index_space();
  return Rect<1>(Runtime::get_runtime()->get_index_space_domain(is));
}

void *OutputRegionArg::untyped_ptr() const
{
#ifdef DEBUG_PANDAS
  assert(nullptr != buffer_);
#endif
  return type_dispatch_primitive_only(code, detail::UntypedPtr{}, buffer_);
}

}  // namespace pandas
}  // namespace legate

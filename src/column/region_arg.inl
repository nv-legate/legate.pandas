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

namespace legate {
namespace pandas {

template <bool READ, int DIM>
RegionArg<READ, DIM>::RegionArg()
  : code(TypeCode::INVALID), pr_(), fid_(-1U), has_rect(false), accessor_(nullptr), domain_()
{
}

template <bool READ, int DIM>
RegionArg<READ, DIM>::RegionArg(TypeCode c, const Legion::PhysicalRegion &pr, Legion::FieldID fid)
  : code(c), pr_(pr), fid_(fid), has_rect(false), accessor_(nullptr), domain_()
{
  if (pr.is_mapped()) {
    has_rect = true;
    domain_  = pr_.operator Legion::DomainT<DIM, Legion::coord_t>();
    if (domain_.dense()) accessor_ = create_accessor<DIM>(code, READ, pr_, fid_);
  } else if (is_meta() && pr.get_logical_region() != Legion::LogicalRegion::NO_REGION) {
    has_rect = true;
    domain_  = pr_.operator Legion::DomainT<DIM, Legion::coord_t>();
  } else
    code = TypeCode::INVALID;
}

template <bool READ, int DIM>
RegionArg<READ, DIM>::~RegionArg()
{
  if (nullptr != accessor_) delete_accessor<DIM>(code, READ, accessor_);
}

template <bool READ, int DIM>
RegionArg<READ, DIM>::RegionArg(RegionArg<READ, DIM> &&other)
  : code(other.code),
    pr_(other.pr_),
    fid_(other.fid_),
    has_rect(other.has_rect),
    accessor_(other.accessor_),
    domain_(other.domain_)
{
  other.accessor_ = nullptr;
}

template <bool READ, int DIM>
RegionArg<READ, DIM> &RegionArg<READ, DIM>::operator=(RegionArg<READ, DIM> &&other)
{
  code            = other.code;
  pr_             = other.pr_;
  fid_            = other.fid_;
  has_rect        = other.has_rect;
  accessor_       = other.accessor_;
  domain_         = other.domain_;
  other.accessor_ = nullptr;
  return *this;
}

template <bool READ, int DIM>
void RegionArg<READ, DIM>::set_rect(const Legion::Rect<DIM> &rect)
{
#ifdef DEBUG_PANDAS
  assert(!has_rect);
#endif
  domain_  = Legion::DomainT<DIM>{rect};
  has_rect = true;
}

template <bool READ, int DIM>
template <typename T>
const AccessorRO<T, DIM> &RegionArg<READ, DIM>::read_accessor(void) const
{
#ifdef DEBUG_PANDAS
  assert(pandas_type_code_of<T> == to_storage_type_code(code));
  assert(valid());
#endif
  return *static_cast<const AccessorRO<T, DIM> *>(accessor_);
}

template <bool READ, int DIM>
template <typename T>
const AccessorWO<T, DIM> &RegionArg<READ, DIM>::write_accessor(void) const
{
#ifdef DEBUG_PANDAS
  assert(pandas_type_code_of<T> == to_storage_type_code(code));
  assert(!READ);
  assert(valid());
#endif
  return *static_cast<const AccessorWO<T, DIM> *>(accessor_);
}

template <bool READ, int DIM>
template <typename T>
T *RegionArg<READ, DIM>::raw_write() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  if (domain_.dense())
    return write_accessor<T>().ptr(domain_.bounds.lo);
  else {
    for (Legion::SpanIterator<WRITE_DISCARD, T, DIM> it(pr_, fid_); it.valid(); ++it)
      return (*it).data();
    assert(false);
  }
  return nullptr;
}

template <bool READ, int DIM>
template <typename T>
const T *RegionArg<READ, DIM>::raw_read() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  if (domain_.dense())
    return read_accessor<T>().ptr(domain_.bounds.lo);
  else {
    for (Legion::SpanIterator<READ_ONLY, T, DIM> it(pr_, fid_); it.valid(); ++it)
      return (*it).data();
    assert(false);
  }
  return nullptr;
}

namespace detail {

template <bool READ, int DIM>
struct RawUntypedWrite {
  template <TypeCode CODE>
  inline void *operator()(const RegionArg<READ, DIM> &region)
  {
    return region.template raw_write<pandas_type_of<CODE>>();
  }
};

template <bool READ, int DIM>
struct RawUntypedRead {
  template <TypeCode CODE>
  inline const void *operator()(const RegionArg<READ, DIM> &region)
  {
    return region.template raw_read<pandas_type_of<CODE>>();
  }
};

}  // namespace detail

template <bool READ, int DIM>
void *RegionArg<READ, DIM>::raw_untyped_write() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  return type_dispatch_primitive_only(code, detail::RawUntypedWrite<READ, DIM>{}, *this);
}

template <bool READ, int DIM>
const void *RegionArg<READ, DIM>::raw_untyped_read() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  return type_dispatch_primitive_only(code, detail::RawUntypedRead<READ, DIM>{}, *this);
}

template <bool READ, int DIM>
size_t RegionArg<READ, DIM>::elem_size() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  return size_of_type(code);
}

template <bool READ, int DIM>
size_t RegionArg<READ, DIM>::bytes() const
{
#ifdef DEBUG_PANDAS
  assert(valid());
#endif
  return size() * elem_size();
}

}  // namespace pandas
}  // namespace legate

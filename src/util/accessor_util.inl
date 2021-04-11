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

template <int DIM>
void *unpack_accessor(LegateDeserializer &derez,
                      TypeCode type_code,
                      bool read,
                      const Legion::PhysicalRegion &region,
                      bool affine /*=true*/)
{
  const Legion::FieldID fid = derez.unpack_32bit_int();
  const int M               = derez.unpack_32bit_int();
  assert(M == 0);
  return create_accessor<DIM>(type_code, read, region, fid, affine);
}

namespace detail {

template <int DIM>
struct CreateAccessor {
  template <TypeCode CODE>
  void *operator()(const Legion::PhysicalRegion &region, Legion::FieldID fid, bool read)
  {
    using T = pandas_type_of<CODE>;
    if (read)
      return new AccessorRO<T, DIM>(region, fid);
    else
      return new AccessorWO<T, DIM>(region, fid);
  }
};

template <int DIM>
struct CreateGenericAccessor {
  template <TypeCode CODE>
  void *operator()(const Legion::PhysicalRegion &region, Legion::FieldID fid, bool read)
  {
    using T = pandas_type_of<CODE>;
    if (read)
      return new GenericAccessorRO<T, DIM>(region, fid);
    else
      return new GenericAccessorWO<T, DIM>(region, fid);
  }
};

template <int DIM>
struct DeleteAccessor {
  template <TypeCode CODE>
  void operator()(void *accessor, bool read)
  {
    using T = pandas_type_of<CODE>;
    if (read)
      delete static_cast<AccessorRO<T, DIM> *>(accessor);
    else
      delete static_cast<AccessorWO<T, DIM> *>(accessor);
  }
};

template <int DIM>
struct DeleteGenericAccessor {
  template <TypeCode CODE>
  void operator()(void *accessor, bool read)
  {
    using T = pandas_type_of<CODE>;
    if (read)
      delete static_cast<GenericAccessorRO<T, DIM> *>(accessor);
    else
      delete static_cast<GenericAccessorWO<T, DIM> *>(accessor);
  }
};

}  // namespace detail

template <int DIM>
void *create_accessor(TypeCode type_code,
                      bool read,
                      const Legion::PhysicalRegion &region,
                      Legion::FieldID fid,
                      bool affine /*=true*/)
{
  if (affine)
    return type_dispatch_primitive_only(
      type_code, detail::CreateAccessor<DIM>{}, region, fid, read);
  else
    return type_dispatch_primitive_only(
      type_code, detail::CreateGenericAccessor<DIM>{}, region, fid, read);
}

template <int DIM>
void delete_accessor(TypeCode type_code, bool read, void *accessor, bool affine /*=true*/)
{
  if (affine)
    return type_dispatch_primitive_only(type_code, detail::DeleteAccessor<DIM>{}, accessor, read);
  else
    return type_dispatch_primitive_only(
      type_code, detail::DeleteGenericAccessor<DIM>{}, accessor, read);
}

}  // namespace pandas
}  // namespace legate

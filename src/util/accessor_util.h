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

#include "util/type_dispatch.h"

namespace legate {
namespace pandas {

template <int DIM = 1>
void *unpack_accessor(LegateDeserializer &derez,
                      TypeCode type_code,
                      bool read,
                      const Legion::PhysicalRegion &region,
                      bool affine = true);

template <int DIM = 1>
void *create_accessor(TypeCode type_code,
                      bool read,
                      const Legion::PhysicalRegion &region,
                      Legion::FieldID fid,
                      bool affine = true);

template <int DIM = 1>
void delete_accessor(TypeCode type_code, bool read, void *accessor, bool affine = true);

}  // namespace pandas
}  // namespace legate

#include "util/accessor_util.inl"

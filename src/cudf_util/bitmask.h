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

#include "bitmask/bitmask.h"
#include <cudf/null_mask.hpp>

namespace legate {
namespace pandas {
namespace util {

void to_bitmask(Bitmask::AllocType* bitmask,
                const Bitmask::AllocType* boolmask,
                size_t num_bits,
                cudaStream_t stream);

void to_boolmask(Bitmask::AllocType* boolmask,
                 const Bitmask::AllocType* bitmask,
                 size_t num_bits,
                 cudaStream_t stream);

}  // namespace util
}  // namespace pandas
}  // namespace legate

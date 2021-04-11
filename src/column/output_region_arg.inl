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

template <typename VAL>
void OutputRegionArg::return_from_buffer(Legion::DeferredBuffer<VAL, 1>& buffer,
                                         size_t num_elements)
{
  out_.return_data(fid_, buffer, &num_elements);
}

template <typename T>
T* OutputRegionArg::ptr() const
{
#ifdef DEBUG_PANDAS
  assert(nullptr != buffer_);
  assert(pandas_type_code_of<T> == to_storage_type_code(code));
#endif
  return static_cast<Legion::DeferredBuffer<T, 1>*>(buffer_)->ptr(0);
}

}  // namespace pandas
}  // namespace legate

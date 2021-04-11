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

#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/strip.hpp>

// Here we define detail methods in cudf that we know are defined but not exposed

namespace cudf {
namespace detail {

std::unique_ptr<column> copy_range(
  column_view const& source,
  column_view const& target,
  size_type source_begin,
  size_type source_end,
  size_type target_begin,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

cudf::size_type count_unset_bits(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

std::unique_ptr<scalar> reduce(
  column_view const& col,
  std::unique_ptr<aggregation> const& agg,
  data_type output_dtype,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> scan(
  const column_view& input,
  std::unique_ptr<aggregation> const& agg,
  scan_type inclusive,
  null_policy null_handling,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<scalar> get_element(column_view const& input,
                                    size_type index,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace cudf

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<column> contains_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> strip(
  strings_column_view const& strings,
  strip_type stype                    = strip_type::BOTH,
  string_scalar const& to_strip       = string_scalar(""),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> to_lower(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> to_upper(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> swapcase(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> zfill(
  strings_column_view const& strings,
  size_type width,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> pad(
  strings_column_view const& strings,
  size_type width,
  pad_side side                       = pad_side::RIGHT,
  std::string const& fill_char        = " ",
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf

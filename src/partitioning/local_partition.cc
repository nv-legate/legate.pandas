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

#include "partitioning/local_partition.h"
#include "partitioning/local_partition_args.h"
#include "column/detail/column.h"
#include "copy/gather.h"
#include "table/row_wrappers.h"
#include "util/allocator.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {

using namespace Legion;
using ColumnView = detail::Column;

namespace partition {

namespace detail {

void hash_all(LocalPartitionArgs &args, std::vector<uint32_t> &radix, std::vector<uint32_t> &hist)
{
  const size_t num_pieces = args.num_pieces;
  const size_t size       = args.input[0].num_elements();

  table::RowHasher hf{};

  std::vector<ColumnView> in_keys;
  for (auto idx : args.key_indices) in_keys.push_back(args.input[idx].view());

  for (size_t i = 0; i < size; ++i) {
    table::Row key{in_keys, i};
    auto r = static_cast<uint32_t>(hf(key) % num_pieces);
    ++hist[r];
    radix[i] = r;
  }
}

static void histogram_to_ranges(LocalPartitionArgs &args,
                                std::vector<int64_t> &offsets,
                                const std::vector<uint32_t> &hist)
{
  // Exclusive sum
  for (size_t i = 1; i < offsets.size(); ++i) offsets[i] = offsets[i - 1] + hist[i - 1];

  const coord_t lo = args.input[0].shape().lo[0];
  const coord_t y  = args.hist_rect.lo[1];
  for (coord_t i = 0; i < args.num_pieces; ++i)
    args.hist_acc[Point<2>(i, y)] = Rect<1>(lo + offsets[i], lo + offsets[i + 1] - 1);
}

static void radix_to_mapping(std::vector<int64_t> &mapping,
                             std::vector<int64_t> &offsets,
                             const std::vector<uint32_t> &radix,
                             const std::vector<uint32_t> &hist,
                             const uint32_t num_pieces)
{
  const size_t size = radix.size();

  for (size_t i = 0; i < size; ++i) mapping[offsets[radix[i]]++] = i;
}

}  // namespace detail

/*static*/ int64_t LocalPartitionTask::cpu_variant(const Task *task,
                                                   const std::vector<PhysicalRegion> &regions,
                                                   Context context,
                                                   Runtime *runtime)
{
  Deserializer ctx{task, regions};

  detail::LocalPartitionArgs args;
  deserialize(ctx, args);

  int64_t size = static_cast<int64_t>(args.input[0].num_elements());
  std::vector<uint32_t> radix(size);
  std::vector<uint32_t> hist(args.num_pieces, 0);
  detail::hash_all(args, radix, hist);

  std::vector<int64_t> offsets(args.num_pieces + 1, 0);
  detail::histogram_to_ranges(args, offsets, hist);

  std::vector<int64_t> mapping(size);
  // Note: detail::radix_to_mapping consumes offsets computed by detail::histogram_to_ranges
  detail::radix_to_mapping(mapping, offsets, radix, hist, args.num_pieces);

  alloc::DeferredBufferAllocator allocator;
  util::for_each(args.output, args.input, [&](auto &output, auto &input) {
    auto &&gathered =
      copy::gather(input.view(), mapping, false, copy::OutOfRangePolicy::IGNORE, allocator);
    output.return_from_view(allocator, gathered);
  });

  return size;
}

static void __attribute__((constructor)) register_tasks(void)
{
  LocalPartitionTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace partition
}  // namespace pandas
}  // namespace legate

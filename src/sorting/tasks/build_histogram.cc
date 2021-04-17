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

#include <numeric>

#include "sorting/tasks/build_histogram.h"
#include "column/detail/column.h"
#include "table/row_wrappers.h"
#include "util/zip_for_each.h"

namespace legate {
namespace pandas {
namespace sorting {

using namespace Legion;

using Table              = BuildHistogramTask::BuildHistogramArgs::Table;
using BuildHistogramArgs = BuildHistogramTask::BuildHistogramArgs;

void BuildHistogramTask::BuildHistogramArgs::sanity_check(void)
{
  for (auto &column : samples) assert(samples[0].shape() == column.shape());
  for (auto &column : input) assert(input[0].shape() == column.shape());
}

static void histogram_to_ranges(BuildHistogramArgs &args,
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

/*static*/ void BuildHistogramTask::cpu_variant(const Task *task,
                                                const std::vector<PhysicalRegion> &regions,
                                                Context context,
                                                Runtime *runtime)
{
  Deserializer ctx{task, regions};

  BuildHistogramArgs args;
  deserialize(ctx, args);

  auto num_samples = args.samples[0].num_elements();

  std::vector<int64_t> sample_indices(num_samples);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  std::vector<detail::Column> sample_views;
  std::vector<detail::Column> input_views;

  for (auto &&sample : args.samples) sample_views.push_back(sample.view());
  for (auto &&input : args.input) input_views.push_back(input.view());

  table::RowCompare op{sample_views, args.ascending, args.put_null_first};
  std::sort(sample_indices.begin(), sample_indices.end(), op);

  auto stride         = num_samples / args.num_pieces;
  auto num_boundaries = args.num_pieces - 1;
  std::vector<int64_t> boundaries(num_boundaries, -1);
  for (auto idx = 0; idx < num_boundaries; ++idx)
    boundaries[idx] = sample_indices[(idx + 1) * stride];

  auto size = args.input[0].num_elements();

  std::vector<uint32_t> hist(args.num_pieces, 0);

  size_t idx = 0;
  for (size_t r = 0; r < num_boundaries; ++r) {
    table::Row boundary{sample_views, static_cast<size_t>(boundaries[r])};
    for (; idx < size; ++idx) {
      table::Row key{input_views, idx};
      if (table::detail::compare_rows(boundary, key, args.ascending, args.put_null_first)) break;
      ++hist[r];
    }
  }

  for (; idx < size; ++idx) ++hist[num_boundaries];

  std::vector<int64_t> offsets(args.num_pieces + 1, 0);
  histogram_to_ranges(args, offsets, hist);
}

void deserialize(Deserializer &ctx, BuildHistogramTask::BuildHistogramArgs &args)
{
  deserialize(ctx, args.num_pieces);
  deserialize(ctx, args.put_null_first);

  uint32_t num_columns = 0;
  deserialize(ctx, num_columns);
  args.ascending.resize(num_columns);
  deserialize(ctx, args.ascending, false);
  args.samples.resize(num_columns);
  deserialize(ctx, args.samples, false);
  args.input.resize(num_columns);
  deserialize(ctx, args.input, false);

  args.hist_rect = deserialize(ctx, args.hist_acc);
}

static void __attribute__((constructor)) register_tasks(void)
{
  BuildHistogramTask::register_variants();
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

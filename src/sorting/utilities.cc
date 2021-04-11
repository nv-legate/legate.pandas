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

#include <unordered_set>
#include <stdlib.h>

#include "realm/timers.h"
#include "sorting/utilities.h"

namespace legate {
namespace pandas {
namespace sorting {

#define MAX_NUM_SAMPLES 32LU

void sample(size_t num_elements, std::vector<int64_t> &samples)
{
  if (num_elements == 0) return;

  auto num_samples = std::min(std::max(num_elements / 4, 1LU), MAX_NUM_SAMPLES);
  std::unordered_set<int64_t> samples_set;

  // FIXME: We use the RNG in stdlib for now
  drand48_data rand_data;
  srand48_r(Realm::Clock::current_time_in_nanoseconds(), &rand_data);

  for (auto i = 0; i < num_samples; ++i) {
    int64_t sample;
    do {
      lrand48_r(&rand_data, &sample);
      sample %= num_elements;
    } while (sample < 0 || samples_set.find(sample) != samples_set.end());
    samples_set.insert(sample);
    samples.push_back(sample);
  }
}

}  // namespace sorting
}  // namespace pandas
}  // namespace legate

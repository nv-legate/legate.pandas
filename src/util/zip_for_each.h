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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace legate {
namespace pandas {
namespace util {

template <typename F>
struct BinaryFunctor {
  template <typename Tuple>
  void operator()(Tuple t)
  {
    func(thrust::get<0>(t), thrust::get<1>(t));
  }
  F func;
};

template <typename Iterator1, typename Iterator2, typename F>
void for_each(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2, F func)
{
  using Pair = thrust::tuple<Iterator1, Iterator2>;
  auto start = thrust::zip_iterator<Pair>(thrust::make_tuple(begin1, begin2));
  auto end   = thrust::zip_iterator<Pair>(thrust::make_tuple(end1, end2));
  thrust::for_each(start, end, BinaryFunctor<F>{func});
}

template <typename Container1, typename Container2, typename F>
void for_each(const Container1& c1, const Container2& c2, F func)
{
  for_each(c1.begin(), c1.end(), c2.begin(), c2.end(), func);
}

template <typename Container1, typename Container2, typename F>
void for_each(Container1& c1, Container2& c2, F func)
{
  for_each(c1.begin(), c1.end(), c2.begin(), c2.end(), func);
}

}  // namespace util
}  // namespace pandas
}  // namespace legate

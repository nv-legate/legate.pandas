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

#include <vector>
#include <unordered_map>

#include "column/detail/column.h"
#include "copy/concatenate.h"
#include "copy/gather.h"
#include "groupby/groupby_reduce.h"
#include "reduction/reduction_op.h"
#include "table/row_wrappers.h"
#include "util/allocator.h"
#include "util/type_dispatch.h"

namespace legate {
namespace pandas {
namespace groupby {

using namespace Legion;
using ColumnsRO   = GroupByReductionTask::GroupByArgs::ColumnsRO;
using ColumnsWO   = GroupByReductionTask::GroupByArgs::ColumnsWO;
using ColumnViews = std::vector<detail::Column>;

using ColumnView = detail::Column;
using TableView  = detail::Table;

void GroupByReductionTask::GroupByArgs::sanity_check(void)
{
  assert(in_keys[0].size() == out_keys.size());
  assert(in_values.size() == all_out_values.size());
  assert(in_keys.size() == in_values[0].size());
  for (unsigned i = 0; i < in_keys.size(); ++i)
    assert(in_keys[i][0].num_elements() == in_values[0][i].num_elements());
  assert(all_out_values.size() == all_aggs.size());
}

namespace detail {

using HashTable = std::unordered_map<table::Row, size_t, table::RowHasher, table::RowEqual>;

void count_keys(HashTable &table, std::vector<int64_t> &gather_map, const TableView &keys)
{
  const auto size = keys.size();

  for (size_t i = 0; i < size; ++i) {
    table::Row key{keys.columns(), i};
    if (!key.all_valid()) continue;
    auto finder = table.find(key);
    if (finder == table.end()) {
      table[key] = gather_map.size();
      gather_map.push_back(i);
    }
  }
}

struct GenericAggregateImpl {
  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::COUNT> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    const auto in_size  = in_value.size();
    const auto out_size = table.size();

    auto out = allocator.allocate_elements<int32_t>(out_size);
    for (size_t i = 0; i < out_size; ++i) out[i] = 0;

    if (in_value.nullable()) {
      auto in_b = in_value.bitmask();

      for (size_t i = 0; i < in_size; ++i) {
        if (!in_b.get(i)) continue;

        table::Row key{in_keys, i};
        if (!key.all_valid()) continue;

        auto finder = table.find(key);
#ifdef DEBUG_PANDAS
        assert(finder != table.end());
#endif
        size_t j = finder->second;
        ++out[j];
      }
    } else {
      for (size_t i = 0; i < in_size; ++i) {
        table::Row key{in_keys, i};
        if (!key.all_valid()) continue;

        auto finder = table.find(key);
#ifdef DEBUG_PANDAS
        assert(finder != table.end());
#endif
        size_t j = finder->second;
        ++out[j];
      }
    }
    return ColumnView{pandas_type_code_of<int32_t>, out, out_size};
  }

  template <AggregationCode AGG, std::enable_if_t<AGG == AggregationCode::SIZE> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    ColumnView non_nullable_in_value{in_value.code(), in_value.raw_column(), in_value.size()};
    return aggregate<AggregationCode::COUNT>(table, in_keys, non_nullable_in_value, allocator);
  }
};

template <AggregationCode AGG, typename VAL>
ColumnView _simple_aggregate(HashTable &table,
                             const ColumnViews &in_keys,
                             ColumnView &in_value,
                             alloc::Allocator &allocator)
{
  using OP_TYPE = reduction::Op<AGG, VAL>;
  using RES     = typename OP_TYPE::result_t;
  OP_TYPE op{};

  const auto in_size  = in_value.size();
  const auto out_size = table.size();

  auto out = allocator.allocate_elements<RES>(out_size);
  for (size_t i = 0; i < out_size; ++i) out[i] = OP_TYPE::identity();

  if (in_value.nullable()) {
    auto raw_out_b = allocator.allocate_elements<Bitmask::AllocType>(out_size);
    Bitmask out_b(raw_out_b, out_size);
    out_b.clear();

    auto in   = in_value.column<VAL>();
    auto in_b = in_value.bitmask();

    for (size_t i = 0; i < in_size; ++i) {
      if (!in_b.get(i)) continue;

      table::Row key{in_keys, i};
      if (!key.all_valid()) continue;

      auto finder = table.find(key);
#ifdef DEBUG_PANDAS
      assert(finder != table.end());
#endif
      size_t j = finder->second;
      out[j]   = op(out[j], in[i]);
      out_b.set(j, true);
    }
    return ColumnView{pandas_type_code_of<RES>, out, out_size, raw_out_b};
  } else {
    auto in = in_value.column<VAL>();

    for (size_t i = 0; i < in_size; ++i) {
      table::Row key{in_keys, i};
      if (!key.all_valid()) continue;

      auto finder = table.find(key);
#ifdef DEBUG_PANDAS
      assert(finder != table.end());
#endif
      size_t j = finder->second;
      out[j]   = op(out[j], in[i]);
    }
    return ColumnView{pandas_type_code_of<RES>, out, out_size};
  }
}

template <TypeCode CODE>
struct NumericAggregateImpl {
  template <AggregationCode AGG,
            std::enable_if_t<is_numeric_type<CODE>::value &&
                             reduction::is_numeric_aggregation<AGG>::value &&
                             !reduction::is_compound_aggregation<AGG>::value> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    return _simple_aggregate<AGG, pandas_type_of<CODE>>(table, in_keys, in_value, allocator);
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<is_numeric_type<CODE>::value && AGG == AggregationCode::MEAN> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    ColumnView sum = aggregate<AggregationCode::SUM>(table, in_keys, in_value, allocator);
    ColumnView cnt =
      GenericAggregateImpl{}.aggregate<AggregationCode::COUNT>(table, in_keys, in_value, allocator);

    auto size = sum.size();
    auto out  = allocator.allocate_elements<double>(size);

    auto p_sum = sum.column<VAL>();
    auto p_cnt = cnt.column<int32_t>();
    for (auto idx = 0; idx < size; ++idx) out[idx] = static_cast<double>(p_sum[idx]) / p_cnt[idx];

    return ColumnView{pandas_type_code_of<double>, out, size, sum.raw_bitmask()};
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<is_numeric_type<CODE>::value && AGG == AggregationCode::VAR> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    using VAL = pandas_type_of<CODE>;

    ColumnView sqsum = aggregate<AggregationCode::SQSUM>(table, in_keys, in_value, allocator);
    ColumnView sum   = aggregate<AggregationCode::SUM>(table, in_keys, in_value, allocator);
    ColumnView cnt =
      GenericAggregateImpl{}.aggregate<AggregationCode::COUNT>(table, in_keys, in_value, allocator);

    auto size  = sum.size();
    auto out   = allocator.allocate_elements<double>(size);
    auto out_b = allocator.allocate_elements<Bitmask::AllocType>(size);

    auto p_sqsum = sqsum.column<VAL>();
    auto p_sum   = sum.column<VAL>();
    auto p_cnt   = cnt.column<int32_t>();
    for (auto idx = 0; idx < size; ++idx) {
      auto cnt    = static_cast<double>(p_cnt[idx]);
      auto sqmean = static_cast<double>(p_sqsum[idx]) / cnt;
      auto mean   = static_cast<double>(p_sum[idx]) / cnt;
      out[idx]    = cnt / (cnt - 1) * (sqmean - mean * mean);
      out_b[idx]  = static_cast<Bitmask::AllocType>(cnt > 1);
    }

    return ColumnView{pandas_type_code_of<double>, out, size, out_b};
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<is_numeric_type<CODE>::value && AGG == AggregationCode::STD> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    ColumnView var = aggregate<AggregationCode::VAR>(table, in_keys, in_value, allocator);
    auto size      = var.size();
    auto out       = allocator.allocate_elements<double>(size);

    auto p_var = var.column<double>();
    for (auto idx = 0; idx < size; ++idx) out[idx] = std::sqrt(p_var[idx]);

    return ColumnView{pandas_type_code_of<double>, out, size, var.raw_bitmask()};
  }

  template <AggregationCode AGG,
            std::enable_if_t<reduction::is_numeric_aggregation<AGG>::value &&
                             !is_numeric_type<CODE>::value> * = nullptr>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

template <TypeCode CODE>
struct MinMaxAggregateImpl {
  template <AggregationCode AGG>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    return _simple_aggregate<AGG, pandas_type_of<CODE>>(table, in_keys, in_value, allocator);
  }
};

template <>
struct MinMaxAggregateImpl<TypeCode::STRING> {
  template <AggregationCode AGG>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    using VAL     = std::string;
    using OP_TYPE = reduction::Op<AGG, VAL>;
    using RES     = typename OP_TYPE::result_t;
    OP_TYPE op{};

    const auto in_size  = in_value.size();
    const auto out_size = table.size();

    if (out_size == 0) return ColumnView(TypeCode::STRING, nullptr, 0);

    std::unordered_map<size_t, std::string> out;

    auto raw_out_b = static_cast<Bitmask::AllocType *>(nullptr);

    if (in_value.nullable()) {
      raw_out_b = allocator.allocate_elements<Bitmask::AllocType>(out_size);
      Bitmask out_b(raw_out_b, out_size);
      out_b.clear();

      auto in_b = in_value.bitmask();

      for (size_t i = 0; i < in_size; ++i) {
        if (!in_b.get(i)) continue;

        table::Row key{in_keys, i};
        if (!key.all_valid()) continue;

        auto finder = table.find(key);
#ifdef DEBUG_PANDAS
        assert(finder != table.end());
#endif
        size_t j = finder->second;
        auto in  = in_value.element<std::string>(i);
        if (out.find(j) == out.end())
          out[j] = in;
        else
          out[j] = op(out[j], in);
        out_b.set(j, true);
      }
    } else {
      auto in = in_value.column<VAL>();

      for (size_t i = 0; i < in_size; ++i) {
        table::Row key{in_keys, i};
        if (!key.all_valid()) continue;

        auto finder = table.find(key);
#ifdef DEBUG_PANDAS
        assert(finder != table.end());
#endif
        size_t j = finder->second;
        auto in  = in_value.element<std::string>(i);
        if (out.find(j) == out.end())
          out[j] = in;
        else
          out[j] = op(out[j], in);
      }
    }

    size_t num_chars = 0;
    for (auto &o : out) num_chars += o.second.size();

    auto offsets     = allocator.allocate_elements<int32_t>(out_size + 1);
    auto chars       = allocator.allocate_elements<int8_t>(num_chars);
    int32_t curr_off = 0;
    for (auto idx = 0; idx < out_size; ++idx) {
      auto &str    = out[idx];
      offsets[idx] = curr_off;
      memcpy(&chars[curr_off], str.c_str(), str.size());
      curr_off += str.size();
    }
    offsets[out_size] = curr_off;

    return ColumnView(TypeCode::STRING,
                      nullptr,
                      out_size,
                      raw_out_b,
                      {ColumnView(TypeCode::INT32, offsets, out_size + 1),
                       ColumnView(TypeCode::INT8, chars, num_chars)});
  }
};

template <>
struct MinMaxAggregateImpl<TypeCode::CAT32> {
  template <AggregationCode AGG>
  ColumnView aggregate(HashTable &table,
                       const ColumnViews &in_keys,
                       ColumnView &in_value,
                       alloc::Allocator &allocator)
  {
    assert(false);
    return ColumnView();
  }
};

template <TypeCode CODE>
struct AggregateDispatch {
  template <AggregationCode AGG,
            std::enable_if_t<reduction::is_numeric_aggregation<AGG>::value> * = nullptr>
  void operator()(HashTable &table,
                  const ColumnViews &in_keys,
                  OutputColumn &out_value,
                  ColumnView &in_value)
  {
    alloc::DeferredBufferAllocator allocator{};
    auto out =
      NumericAggregateImpl<CODE>{}.template aggregate<AGG>(table, in_keys, in_value, allocator);
    out_value.return_from_view(allocator, out);
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<AGG == AggregationCode::COUNT || AGG == AggregationCode::SIZE> * = nullptr>
  void operator()(HashTable &table,
                  const ColumnViews &in_keys,
                  OutputColumn &out_value,
                  ColumnView &in_value)
  {
    alloc::DeferredBufferAllocator allocator{};
    auto out = GenericAggregateImpl{}.template aggregate<AGG>(table, in_keys, in_value, allocator);
    out_value.return_from_view(allocator, out);
  }

  template <
    AggregationCode AGG,
    std::enable_if_t<AGG == AggregationCode::MIN || AGG == AggregationCode::MAX> * = nullptr>
  void operator()(HashTable &table,
                  const ColumnViews &in_keys,
                  OutputColumn &out_value,
                  ColumnView &in_value)
  {
    alloc::DeferredBufferAllocator allocator{};
    auto out =
      MinMaxAggregateImpl<CODE>{}.template aggregate<AGG>(table, in_keys, in_value, allocator);
    out_value.return_from_view(allocator, out);
  }
};

struct TypeDispatch {
  template <TypeCode CODE>
  void operator()(HashTable &table,
                  const ColumnViews &in_keys,
                  const AggregationCode agg,
                  OutputColumn &out_value,
                  ColumnView &in_value)
  {
    type_dispatch(agg, AggregateDispatch<CODE>{}, table, in_keys, out_value, in_value);
  }
};

void aggregate(HashTable &table,
               const ColumnViews &in_keys,
               const AggregationCode agg,
               OutputColumn &out_value,
               ColumnView &in_value)
{
  type_dispatch(in_value.code(), TypeDispatch{}, table, in_keys, agg, out_value, in_value);
}

}  // namespace detail

/*static*/ int64_t GroupByReductionTask::cpu_variant(const Task *task,
                                                     const std::vector<PhysicalRegion> &regions,
                                                     Context context,
                                                     Runtime *runtime)
{
  Deserializer des(task, regions);
  GroupByArgs args;
  deserialize(des, args);

  detail::HashTable table;
  std::vector<int64_t> gather_map;

  uint32_t num_inputs = args.in_keys.size();

  std::vector<TableView> all_key_views;
  for (auto &keys : args.in_keys) {
    ColumnViews views;
    for (auto &&key : keys) views.push_back(key.view());
    all_key_views.push_back(TableView{std::move(views)});
  }

  alloc::DeferredBufferAllocator allocator;
  TableView in_keys = copy::concatenate(all_key_views, allocator);

  detail::count_keys(table, gather_map, in_keys);

  int64_t out_size = static_cast<int64_t>(gather_map.size());

  if (out_size == 0) {
    for (auto &out_key : args.out_keys) out_key.make_empty(true);
    for (auto &out_values : args.all_out_values)
      for (auto &out_value : out_values) out_value.make_empty(true);
    return 0;
  }

  const size_t num_groups = args.all_out_values.size();
  for (size_t grp_idx = 0; grp_idx < num_groups; ++grp_idx) {
    ColumnViews in_values;
    for (auto &in_value : args.in_values[grp_idx]) in_values.push_back(in_value.view());
    auto in_value = copy::concatenate(in_values, allocator);

    size_t num_aggs = args.all_out_values[grp_idx].size();
    for (size_t agg_idx = 0; agg_idx < num_aggs; ++agg_idx) {
      detail::aggregate(table,
                        in_keys.columns(),
                        args.all_aggs[grp_idx][agg_idx],
                        args.all_out_values[grp_idx][agg_idx],
                        in_value);
    }
  }

  for (auto idx = 0; idx < in_keys.num_columns(); ++idx) {
    auto out_key = copy::gather(
      in_keys.column(idx), gather_map, false, copy::OutOfRangePolicy::IGNORE, allocator);
    args.out_keys[idx].return_from_view(allocator, out_key);
  }

  return out_size;
}

void deserialize(Deserializer &des, GroupByReductionTask::GroupByArgs &args)
{
  uint32_t num_keys = 0;
  deserialize(des, num_keys);  // # of key columns

  uint32_t num_inputs = 0;
  deserialize(des, num_inputs);  // maximum # of inputs

  for (uint32_t key_idx = 0; key_idx < num_inputs; ++key_idx) {
    ColumnsRO columns(num_keys);
    deserialize(des, columns, false);
    bool all_valid = true;
    for (auto &column : columns) all_valid = all_valid && column.valid();
    if (all_valid) args.in_keys.push_back(std::move(columns));
  }

  args.out_keys.resize(num_keys);
  deserialize(des, args.out_keys, false);

  uint32_t num_groups = 0;
  deserialize(des, num_groups);  // # of value columns

  args.all_aggs.resize(num_groups);
  args.all_out_values.resize(num_groups);
  for (uint32_t grp_idx = 0; grp_idx < num_groups; ++grp_idx) {
    uint32_t num_aggs = 0;
    deserialize(des, num_aggs);

    args.all_aggs[grp_idx].resize(num_aggs);
    deserialize(des, args.all_aggs[grp_idx], false);

    args.all_out_values[grp_idx].resize(num_aggs);
    deserialize(des, args.all_out_values[grp_idx], false);

    uint32_t num_inputs = 0;
    deserialize(des, num_inputs);  // maximum # of inputs
    ColumnsRO in_columns;
    for (uint32_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
      Column<true> column;
      deserialize(des, column);
      if (column.valid()) in_columns.push_back(std::move(column));
    }
    args.in_values.push_back(std::move(in_columns));
  }

#ifdef DEBUG_PANDAS
  args.sanity_check();
#endif
}

static void __attribute__((constructor)) register_tasks(void)
{
  GroupByReductionTask::register_variants_with_return<int64_t, int64_t>();
}

}  // namespace groupby
}  // namespace pandas
}  // namespace legate

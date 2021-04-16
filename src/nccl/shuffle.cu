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

#include "nccl/shuffle.h"
#include "nccl/util.h"
#include "util/allocator.h"
#include "util/cuda_helper.h"

#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>

namespace legate {
namespace pandas {
namespace comm {

using namespace Legion;

namespace detail {

struct packed_columns_view {
  packed_columns_view(const cudf::packed_columns &columns)
    : metadata(columns.metadata_.get()), gpu_data(columns.gpu_data.get())
  {
  }
  cudf::packed_columns::metadata *metadata;
  rmm::device_buffer *gpu_data;
};

std::unique_ptr<cudf::table> shuffle(const std::vector<packed_columns_view> &packed_tables,
                                     coord_t task_id,
                                     coord_t num_pieces,
                                     ncclComm_t *comm,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource *mr)
{
  // Use the default resource for temporary allocations
  auto temp_mr = rmm::mr::get_current_device_resource();

  AutoStream as{};
  auto meta_stream = as.stream();

  // All-gather buffer sizes so that receivers can allocate buffers of the right sizes
  Rect<1> rect(Point<1>(0), Point<1>(num_pieces * num_pieces - 1));
  DeferredBuffer<size_t, 1> all_buffer_sizes(Memory::Z_COPY_MEM, rect);
  DeferredBuffer<size_t, 1> all_metadata_sizes(Memory::Z_COPY_MEM, rect);

  auto get_aligned_size = [](size_t size) { return std::max<size_t>(16, (size + 15) / 16 * 16); };
  auto index = [&num_pieces](coord_t from, coord_t to) { return from * num_pieces + to; };
  for (coord_t to = 0; to < num_pieces; ++to) {
    auto &table             = packed_tables[to];
    auto idx                = index(task_id, to);
    all_buffer_sizes[idx]   = table.gpu_data->size();
    all_metadata_sizes[idx] = table.metadata->size();
  }

  NCCLCHECK(ncclAllGather(all_buffer_sizes.ptr(task_id * num_pieces),
                          all_buffer_sizes.ptr(0),
                          num_pieces,
                          ncclUint64,
                          *comm,
                          meta_stream));
  NCCLCHECK(ncclAllGather(all_metadata_sizes.ptr(task_id * num_pieces),
                          all_metadata_sizes.ptr(0),
                          num_pieces,
                          ncclUint64,
                          *comm,
                          meta_stream));

  // We must synchronize here before proceeding, as we need the sizes to arrive in order to allocate
  // buffers below
  SYNC_AND_CHECK_STREAM(meta_stream);

  // Allocate necessary buffers for the trasnfers
  std::vector<std::unique_ptr<rmm::device_buffer>> recv_data_buffers;
  for (coord_t from = 0; from < num_pieces; ++from) {
    auto buffer_size = get_aligned_size(all_buffer_sizes[index(from, task_id)]);
    recv_data_buffers.push_back(std::make_unique<rmm::device_buffer>(buffer_size, stream, temp_mr));
  }

  auto meta_buffer_allocator      = alloc::DeferredBufferAllocator(Memory::Kind::GPU_FB_MEM);
  auto host_meta_buffer_allocator = alloc::DeferredBufferAllocator(Memory::Kind::Z_COPY_MEM);
  std::vector<uint8_t *> send_meta_buffers;
  std::vector<uint8_t *> recv_meta_buffers;
  for (coord_t other_id = 0; other_id < num_pieces; ++other_id) {
    auto send_size = get_aligned_size(all_metadata_sizes[index(task_id, other_id)]);
    auto recv_size = get_aligned_size(all_metadata_sizes[index(other_id, task_id)]);

    auto send_buffer = static_cast<uint8_t *>(meta_buffer_allocator.allocate(send_size));
    auto recv_buffer = static_cast<uint8_t *>(meta_buffer_allocator.allocate(recv_size));

    auto &send_metadata = packed_tables[other_id].metadata;
    cudaMemcpyAsync(send_buffer,
                    send_metadata->data(),
                    send_metadata->size(),
                    cudaMemcpyHostToDevice,
                    meta_stream);

    send_meta_buffers.push_back(send_buffer);
    recv_meta_buffers.push_back(recv_buffer);
  }

  // Perform all-to-all exchange. We exchange the host-sidemetadata first
  // so that we can run the unpacking logic while the data is being transferred.
  NCCLCHECK(ncclGroupStart());
  for (auto other_id = 0; other_id < num_pieces; ++other_id) {
    auto send_size = get_aligned_size(all_metadata_sizes[index(task_id, other_id)]);
    auto recv_size = get_aligned_size(all_metadata_sizes[index(other_id, task_id)]);

    auto send_buffer = send_meta_buffers[other_id];
    auto recv_buffer = recv_meta_buffers[other_id];

    NCCLCHECK(ncclSend(send_buffer, send_size, ncclInt8, other_id, *comm, meta_stream));
    NCCLCHECK(ncclRecv(recv_buffer, recv_size, ncclInt8, other_id, *comm, meta_stream));
  }
  NCCLCHECK(ncclGroupEnd());

  std::vector<uint8_t *> host_recv_meta_buffers;
  for (coord_t other_id = 0; other_id < num_pieces; ++other_id) {
    auto recv_size        = get_aligned_size(all_metadata_sizes[index(other_id, task_id)]);
    auto recv_buffer      = recv_meta_buffers[other_id];
    auto host_recv_buffer = static_cast<uint8_t *>(host_meta_buffer_allocator.allocate(recv_size));

    cudaMemcpyAsync(host_recv_buffer, recv_buffer, recv_size, cudaMemcpyDeviceToHost, meta_stream);
    host_recv_meta_buffers.push_back(host_recv_buffer);
  }

  rmm::device_buffer dummy_buffer(get_aligned_size(0), stream, temp_mr);
  NCCLCHECK(ncclGroupStart());
  for (auto other_id = 0; other_id < num_pieces; ++other_id) {
    auto send_size = get_aligned_size(all_buffer_sizes[index(task_id, other_id)]);
    auto recv_size = get_aligned_size(all_buffer_sizes[index(other_id, task_id)]);

    auto send_buffer = packed_tables[other_id].gpu_data->data();
    send_buffer      = nullptr == send_buffer ? dummy_buffer.data() : send_buffer;
    auto recv_buffer = recv_data_buffers[other_id]->data();

    NCCLCHECK(ncclSend(send_buffer, send_size, ncclInt8, other_id, *comm, stream));
    NCCLCHECK(ncclRecv(recv_buffer, recv_size, ncclInt8, other_id, *comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

#ifdef DEBUG_PANDAS
  SYNC_AND_CHECK_STREAM(stream);
#endif

  // This synchronization is mandatory, since the unpacking needs the host-side metadata
  SYNC_AND_CHECK_STREAM(meta_stream);

  // Once we've received all tables (more precisely, their metadata), unpack and concatenate them
  // to construct the result.
  std::vector<cudf::table_view> tables;
  for (auto other_id = 0; other_id < num_pieces; ++other_id) {
    auto buffer_size = all_buffer_sizes[index(other_id, task_id)];
    auto data_buffer =
      buffer_size > 0 ? static_cast<const uint8_t *>(recv_data_buffers[other_id]->data()) : nullptr;
    tables.push_back(cudf::unpack(host_recv_meta_buffers[other_id], data_buffer));
  }

  return cudf::detail::concatenate(tables, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> shuffle(const cudf::table_view &input,
                                     const std::vector<cudf::size_type> &splits,
                                     coord_t task_id,
                                     ncclComm_t *comm,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource *mr)
{
  // Use the default resource for temporary allocations
  auto temp_mr = rmm::mr::get_current_device_resource();

  // Split the table into contiguous chunks
  auto packed_subtables = contiguous_split(input, splits, stream, temp_mr);

  std::vector<detail::packed_columns_view> packed_subtables_views;
  for (auto &packed_subtable : packed_subtables)
    packed_subtables_views.emplace_back(packed_subtable.data);

  return detail::shuffle(packed_subtables_views, task_id, splits.size() + 1, comm, stream, mr);
}

std::unique_ptr<cudf::table> all_gather(const cudf::table_view &input,
                                        coord_t task_id,
                                        coord_t num_tasks,
                                        ncclComm_t *comm,
                                        cudaStream_t stream,
                                        rmm::mr::device_memory_resource *mr)
{
  // Use the default resource for temporary allocations
  auto temp_mr = rmm::mr::get_current_device_resource();

  // Split the table into contiguous chunks
  auto packed_table = pack(input, stream, temp_mr);

  std::vector<detail::packed_columns_view> broadcasted_tables(num_tasks, packed_table);
  return detail::shuffle(broadcasted_tables, task_id, num_tasks, comm, stream, mr);
}

std::pair<cudf::table_view, std::unordered_map<uint32_t, cudf::column_view>> extract_dictionaries(
  const cudf::table_view &input)
{
  std::vector<cudf::column_view> columns;
  std::unordered_map<uint32_t, cudf::column_view> dictionaries;

  for (auto idx = 0; idx < input.num_columns(); ++idx) {
    auto &column = input.column(idx);
    if (column.type().id() == cudf::type_id::DICTIONARY32) {
      const auto &codes = column.child(0);
      dictionaries[idx] = column.child(1);
      columns.push_back(
        cudf::column_view(codes.type(), column.size(), codes.head(), column.null_mask()));
    } else
      columns.push_back(column);
  }

  return std::make_pair(cudf::table_view(std::move(columns)), std::move(dictionaries));
}

std::pair<std::unique_ptr<cudf::table>, std::unordered_map<uint32_t, std::unique_ptr<cudf::column>>>
extract_dictionaries(std::unique_ptr<cudf::table> &&input)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::unordered_map<uint32_t, std::unique_ptr<cudf::column>> dictionaries;

  auto input_columns = input->release();
  for (auto idx = 0; idx < input_columns.size(); ++idx) {
    auto &column = input_columns[idx];
    if (column->type().id() == cudf::type_id::DICTIONARY32) {
      // Hash out the column
      auto size     = column->size();
      auto contents = column->release();

      // Set aside the dictionary that will later be merged back to the column
      dictionaries[idx] = std::move(contents.children[1]);

      // Combine the parent's bitmask with the codes column and put it in the table
      auto &codes         = contents.children[0];
      auto codes_contents = codes->release();

      auto combined = std::make_unique<cudf::column>(
        codes->type(), size, std::move(*codes_contents.data), std::move(*codes_contents.null_mask));
      columns.push_back(std::move(combined));
    } else
      columns.push_back(std::move(column));
  }

  return std::make_pair(std::make_unique<cudf::table>(std::move(columns)), std::move(dictionaries));
}

cudf::table_view embed_dictionaries(
  const cudf::table_view &input,
  const std::unordered_map<uint32_t, cudf::column_view> &dictionaries)
{
  std::vector<cudf::column_view> columns;
  for (uint32_t idx = 0; idx < input.num_columns(); ++idx) {
    auto &column = input.column(idx);

    auto finder = dictionaries.find(idx);
    if (finder == dictionaries.end())
      columns.push_back(column);
    else {
      auto codes = cudf::column_view(column.type(), column.size(), column.head());
      columns.push_back(cudf::column_view(cudf::data_type(cudf::type_id::DICTIONARY32),
                                          column.size(),
                                          nullptr,
                                          column.null_mask(),
                                          -1,
                                          0,
                                          {codes, finder->second}));
    }
  }
  return cudf::table_view(std::move(columns));
}

std::unique_ptr<cudf::table> embed_dictionaries(
  std::unique_ptr<cudf::table> &&input,
  const std::unordered_map<uint32_t, cudf::column_view> &dictionaries)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  auto input_columns = input->release();
  for (uint32_t idx = 0; idx < input_columns.size(); ++idx) {
    auto &column = input_columns[idx];

    auto finder = dictionaries.find(idx);
    if (finder == dictionaries.end())
      columns.push_back(std::move(column));
    else {
      auto codes_size     = column->size();
      auto codes_contents = column->release();

      std::vector<std::unique_ptr<cudf::column>> children;
      children.push_back(std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::UINT32), codes_size, std::move(*codes_contents.data)));

      columns.push_back(std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::DICTIONARY32),
                                                       codes_size,
                                                       rmm::device_buffer{},
                                                       std::move(*codes_contents.null_mask),
                                                       -1,
                                                       std::move(children)));
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace comm
}  // namespace pandas
}  // namespace legate

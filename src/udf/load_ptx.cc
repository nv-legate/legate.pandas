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

#include "udf/load_ptx.h"
#include <regex>
#include <cuda.h>

namespace legate {
namespace pandas {
namespace udf {

using namespace Legion;

/*static*/ uint64_t LoadPTXTask::gpu_variant(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context context,
                                             Runtime *runtime)
{
#ifdef DEBUG_PANDAS
  assert(task->futures.size() == 1);
#endif
  Deserializer ctx{task, regions};
  FromRawFuture future;
  deserialize(ctx, future);
  auto ptx = static_cast<const char *>(future.rawptr_);

  const unsigned num_options = 4;
  const size_t buffer_size   = 16384;
  std::vector<char> log_info_buffer(buffer_size);
  std::vector<char> log_error_buffer(buffer_size);
  CUjit_option jit_options[] = {
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  void *option_vals[] = {
    static_cast<void *>(log_info_buffer.data()),
    reinterpret_cast<void *>(buffer_size),
    static_cast<void *>(log_error_buffer.data()),
    reinterpret_cast<void *>(buffer_size),
  };

  CUmodule module;
  CUresult result = cuModuleLoadDataEx(&module, ptx, num_options, jit_options, option_vals);
  if (result != CUDA_SUCCESS) {
    if (result == CUDA_ERROR_OPERATING_SYSTEM) {
      fprintf(stderr,
              "ERROR: Device side asserts are not supported by the "
              "CUDA driver for MAC OSX, see NVBugs 1628896.\n");
      exit(-1);
    } else if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
      fprintf(stderr, "ERROR: The binary was compiled for the wrong GPU architecture.\n");
      exit(-1);
    } else {
      fprintf(stderr, "Failed to load CUDA module! Error log: %s\n", log_error_buffer.data());
#if CUDA_VERSION >= 6050
      const char *name, *str;
      assert(cuGetErrorName(result, &name) == CUDA_SUCCESS);
      assert(cuGetErrorString(result, &str) == CUDA_SUCCESS);
      fprintf(stderr, "CU: cuModuleLoadDataEx = %d (%s): %s\n", result, name, str);
#else
      fprintf(stderr, "CU: cuModuleLoadDataEx = %d\n", result);
#endif
      exit(-1);
    }
  }

  std::cmatch line_match;
  bool match = std::regex_search(ptx, line_match, std::regex(".visible .entry [_a-zA-Z0-9$]+"));
#ifdef DEBUG_PANDAS
  assert(match);
#endif
  const auto &matched_line = line_match.begin()->str();
  auto fun_name            = matched_line.substr(matched_line.rfind(" ") + 1, matched_line.size());

  CUfunction hfunc;
  result = cuModuleGetFunction(&hfunc, module, fun_name.c_str());
  assert(result == CUDA_SUCCESS);
  return reinterpret_cast<uint64_t>(hfunc);
}

static void __attribute__((constructor)) register_tasks(void)
{
  LoadPTXTask::register_variants_with_return<uint64_t, uint64_t>();
}

}  // namespace udf
}  // namespace pandas
}  // namespace legate

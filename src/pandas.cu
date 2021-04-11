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

#ifdef LEGATE_USE_CUDA

extern "C" {

unsigned legate_pandas_get_cuda_arch()
{
#ifdef PANDAS_DYNAMIC_CUDA_ARCH
  return -1U;
#else

#ifdef FERMI_ARCH
  return 20;
#elif KEPLER_ARCH
  return 30;
#elif K20_ARCH
  return 35;
#elif K80_ARCH
  return 37;
#elif MAXWELL_ARCH
  return 52;
#elif PASCAL_ARCH
  return 60;
#elif VOLTA_ARCH
  return 70;
#elif TURING_ARCH
  return 75;
#elif AMPERE_ARCH
  return 80;
#else
#error "Unable to detect the CUDA architecture"
#endif

#endif
}

bool legate_pandas_use_nccl()
{
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}
}

#endif

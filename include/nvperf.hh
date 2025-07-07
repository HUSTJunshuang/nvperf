/*
 * Copyright 2025 HUSTJunshuang
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
 */

#ifndef __NVPERF_HH__
#define __NVPERF_HH__

#include <assert.h>
#include <cuda/std/chrono>
#include <type_traits>
#include <vector>

namespace cuda_chrono = cuda::std::chrono;

#define NVPERF_EVENT(name) __device__ double *name##_timestamp_d = nullptr

#define NVPERF_EVENT_INIT(name, grid_size) \
    static int name##_grid_size = (grid_size);                                      \
    double *name##_timestamp_h = nullptr;                                           \
    cudaMalloc(&name##_timestamp_h, 2 * (grid_size) * sizeof(double));              \
    cudaMemcpyToSymbol(name##_timestamp_d, &name##_timestamp_h, sizeof(double *))

#define NVPERF_EVENT_START(name) \
    int name##_tid = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;   \
    int name##_bid = blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x;        \
    __syncthreads();                                                                        \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_start = cuda_chrono::high_resolution_clock::now()

#define NVPERF_EVENT_END(name) \
    __syncthreads();    \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_end = cuda_chrono::high_resolution_clock::now(); \
    if (name##_tid == 0 && name##_timestamp_d != nullptr) { \
        name##_timestamp_d[name##_bid * 2] = name##_start.time_since_epoch().count() / 1000.0;  \
        name##_timestamp_d[name##_bid * 2 + 1] = name##_end.time_since_epoch().count() / 1000.0;    \
    }

#define NVPERF_EVENT_ELAPSED(name, out_vector) \
    static_assert(std::is_same<typename std::decay<decltype(out_vector)>::type, std::vector<double> >::value, \
        "[NVPERF_EVENT_ELAPSED] Error: '" #out_vector "' must be of type std::vector<double>."); \
    assert(out_vector.size() == 2 * name##_grid_size);   \
    cudaMemcpy(out_vector.data(), name##_timestamp_h, 2 * name##_grid_size * sizeof(double), cudaMemcpyDeviceToHost)

#define NVPERF_EVENT_DESTROY(name) \
    cudaFree(name##_timestamp_h)

#endif // __NVPERF_HH__
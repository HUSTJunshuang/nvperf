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
#include <cstdio>
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


#define NVPERF_DEVICE_TIMER(name) __device__ double *name##_timer_d = nullptr

#define NVPERF_DEVICE_TIMER_INIT(name, grid_size) \
    int name##_timer_grid_size = (grid_size);                               \
    double *name##_timer_h = nullptr;                                       \
    cudaMalloc(&name##_timer_h, (grid_size) * sizeof(double));              \
    cudaMemset(name##_timer_h, 0.0, (grid_size) * sizeof(double));          \
    cudaMemcpyToSymbol(name##_timer_d, &name##_timer_h, sizeof(double *))

#define NVPERF_DEVICE_TIMER_PUSH(name) \
    int name##_timer_tid = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x; \
    int name##_timer_bid = blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x;      \
    __syncthreads();                                                                            \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_timer_start = cuda_chrono::high_resolution_clock::now()

#define NVPERF_DEVICE_TIMER_POP(name) \
    __syncthreads();                                                                                                                                        \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_timer_end = cuda_chrono::high_resolution_clock::now();                               \
    if (name##_timer_tid == 0 && name##_timer_d != nullptr) {                                                                                               \
        name##_timer_d[name##_timer_bid] += cuda_chrono::duration_cast<cuda_chrono::nanoseconds>(name##_timer_end - name##_timer_start).count() / 1000.0;   \
    }

#define NVPERF_DEVICE_TIMER_ELAPSED(name, out_vector) \
    static_assert(std::is_same<typename std::decay<decltype(out_vector)>::type, std::vector<double>>::value,    \
        "[NVPERF_DEVICE_TIMER_ELAPSED]: '" #out_vector "' must be of type std::vector<double>.");               \
    assert(out_vector.size() >= name##_timer_grid_size);                                                        \
    cudaMemcpy(out_vector.data(), name##_timer_h, name##_timer_grid_size * sizeof(double), cudaMemcpyDeviceToHost)

#define NVPERF_DEVICE_TIMER_DISPLAY(name, out_vector) \
    static_assert(std::is_same<typename std::decay<decltype(out_vector)>::type, std::vector<double>>::value,    \
        "[NVPERF_DEVICE_TIMER_DISPLAY]: '" #out_vector "' must be of type std::vector<double>.");               \
    assert(out_vector.size() >= name##_timer_grid_size);                                                        \
    do {                                                                                                        \
        int active_block = 0;                                                                                   \
        double min_duration = out_vector[0], max_duration = 0.0, aver_duration = 0.0;                           \
        for (int cur_bid = 0; cur_bid < name##_timer_grid_size; ++cur_bid) {                                    \
            if (out_vector[cur_bid] == 0.0) continue;                                                           \
            min_duration = (out_vector[cur_bid] < min_duration) ? out_vector[cur_bid] : min_duration;           \
            max_duration = (out_vector[cur_bid] > max_duration) ? out_vector[cur_bid] : max_duration;           \
            aver_duration += out_vector[cur_bid];                                                               \
            ++active_block;                                                                                     \
        }                                                                                                       \
        aver_duration /= active_block;                                                                          \
        printf(#name " cost %.5lf us (min = %.5lf, max = %.5lf)\n", aver_duration, min_duration, max_duration); \
    } while(0) 

#define NVPERF_DEVICE_TIMER_DESTROY(name) cudaFree(name##_timer_h)


#define NVPERF_HOST_TIMER(name) \
    double name##_host_timer = 0.0;                                                         \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_host_timer_start{};  \
    cuda_chrono::time_point<cuda_chrono::high_resolution_clock> name##_host_timer_end{}

#define NVPERF_HOST_TIMER_PUSH(name) name##_host_timer_start = cuda_chrono::high_resolution_clock::now()

#define NVPERF_HOST_TIMER_POP(name) \
    name##_host_timer_end = cuda_chrono::high_resolution_clock::now();  \
    name##_host_timer += cuda_chrono::duration_cast<cuda_chrono::nanoseconds>(name##_host_timer_end - name##_host_timer_start).count() / 1000.0

#define NVPERF_HOST_TIMER_ELAPSED(name) (name##_host_timer)

#define NVPERF_HOST_TIMER_DISPLAY(name) printf(#name " cost %.5lf us\n", name##_host_timer)


#define NVPERF_MACRO_OVERLOAD_2(pos1, pos2, FUNC, ...) FUNC


#define NVPERF_HOST_ASYNC_TIMER(name) \
    float name##_host_async_timer = 0.0;                \
    cudaEvent_t name##_event_start, name##_event_end;   \
    cudaEventCreate(&name##_event_start);               \
    cudaEventCreate(&name##_event_end)    

#define NVPERF_HOST_ASYNC_TIMER_PUSH_0(name) cudaEventRecord(name##_event_start, cudaStreamDefault)

#define NVPERF_HOST_ASYNC_TIMER_PUSH_1(name, stream) \
    static_assert(std::is_same<typename std::decay<decltype(stream)>::type, cudaStream_t>::value,   \
        "[NVPERF_HOST_ASYNC_TIMER_PUSH]: '" #stream "' must be of type cudaStream_t.");             \
    cudaEventRecord(name##_event_start, stream)

#define NVPERF_HOST_ASYNC_TIMER_PUSH(...) \
    NVPERF_MACRO_OVERLOAD_2(__VA_ARGS__, NVPERF_HOST_ASYNC_TIMER_PUSH_1, NVPERF_HOST_ASYNC_TIMER_PUSH_0)(__VA_ARGS__)

#define NVPERF_HOST_ASYNC_TIMER_POP_0(name) cudaEventRecord(name##_event_end, cudaStreamDefault)

#define NVPERF_HOST_ASYNC_TIMER_POP_1(name, stream) \
    static_assert(std::is_same<typename std::decay<decltype(stream)>::type, cudaStream_t>::value,   \
        "[NVPERF_HOST_ASYNC_TIMER_POP]: '" #stream "' must be of type cudaStream_t.");              \
    cudaEventRecord(name##_event_end, stream);                                                      \

#define NVPERF_HOST_ASYNC_TIMER_POP(...) \
    NVPERF_MACRO_OVERLOAD_2(__VA_ARGS__, NVPERF_HOST_ASYNC_TIMER_POP_1, NVPERF_HOST_ASYNC_TIMER_POP_0)(__VA_ARGS__)

// TODO 改成HOST_EVENT得了
#define NVPERF_HOST_ASYNC_TIMER_ELAPSED(name, duration) \
    static_assert(std::is_same<typename std::decay<decltype(duration)>::type, float>::value,        \
        "[NVPERF_HOST_ASYNC_TIMER_ELAPSED]: '" #duration "' must be of type float.");        \
    cudaEventSynchronize(name##_event_end);                                                         \
    do {                                                                                            \
        cudaError_t err = cudaEventElapsedTime(&duration, name##_event_start, name##_event_end);    \
        if (err != cudaSuccess) {                                                                   \
            printf("[NVPERF_HOST_ASYNC_TIMER_ELAPSED] %s:%d: Error %d: %s, maybe you used ASYNC_TIMER of " #name " in different CUDA Streams.", err, cudaGetErrorString(err));  \
        }                                                                                           \
    } while(0)
    
#define NVPERF_HOST_ASYNC_TIMER_DISPLAY(name) printf(#name "cost %.5lf ms\n", name##_host_async_timer)

#define NVPERF_HOST_ASYNC_TIMER_DESTROY(name)   \
    cudaEventDestroy(name##_event_start);       \
    cudaEventDestroy(name##_event_end)

#endif // __NVPERF_HH__
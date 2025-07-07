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

#include <stdio.h>
#include <nvperf.hh>
#include <vector>

__global__ void add_kernel(float *x, float *y, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] + y[i];
    }
}

__global__ void vadd_kernel(float *x, float *y, float *out, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

NVPERF_EVENT(gadd);

__global__ void gadd_kernel(float *x, float *y, float *out, int n) {
    NVPERF_EVENT_START(gadd);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // for time measurement
    int repeat_cnt = 100000;
    if (tid < n) {
        while (repeat_cnt--)
        out[tid] = x[tid] + y[tid];
    }
    NVPERF_EVENT_END(gadd);
}

int main() {
    int N = 16777216;
    size_t mem_size = N * sizeof(float);
    float *x, *y, *out;
    float *cuda_x, *cuda_y, *cuda_out;
    // allocate host mem
    x = (float *)malloc(mem_size);
    y = (float *)malloc(mem_size);
    out = (float *)malloc(mem_size);
    // init data
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0;
        y[i] = 2.0;
    }
    // allocate device mem and transfer
    cudaMalloc((void **)&cuda_x, mem_size);cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&cuda_y, mem_size);cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&cuda_out, mem_size);    
    
    // lauch add kernel
	add_kernel<<<1, 1>>>(cuda_x, cuda_y, cuda_out, N);                      // single thread
	vadd_kernel<<<1, 256>>>(cuda_x, cuda_y, cuda_out, N);                   // 256 threads
	int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;
    NVPERF_EVENT_INIT(gadd, grid_size);
	gadd_kernel<<<grid_size, block_size>>>(cuda_x, cuda_y, cuda_out, N);    // one calculation per thread
    std::vector<double> timestamp(grid_size * 2);
    NVPERF_EVENT_ELAPSED(gadd, timestamp);
    double min_duration = timestamp[1] - timestamp[0], max_duration = 0.0, aver_duration = 0.0;
    double lauch_time = timestamp[0], end_time = timestamp[1];
    for (int i = 0; i < grid_size; ++i) {
        lauch_time = min(lauch_time, timestamp[i * 2]);
        end_time = max(end_time, timestamp[i * 2 + 1]);
        double duration = timestamp[i * 2 + 1] - timestamp[i * 2];
        min_duration = (duration < min_duration) ? duration : min_duration;
        max_duration = (duration > max_duration) ? duration : max_duration;
        aver_duration += duration;
    }
    aver_duration /= grid_size;
    printf("gadd start @ %.2lf, end @ %.2lf, duration = %.2lf us\n", lauch_time, end_time, end_time - lauch_time);
    printf("gadd cost %.5lf us(min = %.5lf, max = %.5lf)\n", aver_duration, min_duration, max_duration);
    // copy result back
    cudaMemcpy(out, cuda_out, mem_size, cudaMemcpyDeviceToHost);
    // Sync CUDA stream to wait kernel end
    cudaDeviceSynchronize();
    // printf top ten result to check answer
    for (int i = 0; i < 10; ++i) {
        printf("out[%d] = %.3f\n", i, out[i]);
    }
    // free allocated mem
    cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_out);
    free(x);
    free(y);
    free(out);
    NVPERF_EVENT_DESTROY(gadd);

    return 0;
}

# nvperf

## 前言

在测试CUDA代码性能的过程中，我们时常需要测试kernel函数内部某些代码片段的耗时，定位其中的热点代码片段。

但Nsight Compute的源码级分析操作较复杂，在分析大型代码时耗时较长，且只能查看每一行代码执行的指令条数，无法看到每行代码具体执行了多长时间，于是基于CUDA的[chrono库](https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time)设计了nvperf，用于快速得到较为精确的结果。

## nvperf使用方法

### 事件计时

事件计时方式采取记录代码调用时的时间戳的方式进行计时，因此要求被测代码段是不可重入的，否则结果将会被最后一次运行结果覆盖。事件计时的优点是保留了时间戳信息，可以可视化所有block的执行历程（TODO）。

事件计时使用流程如下：

1. 待测代码片段所在文件的全局作用域中调用 `NVPERF_EVENT()`注册事件；
2. host代码中调用 `NVPERF_EVENT_INIT()`初始化所需的变量及内存等；
3. kernel函数内待测代码片段前后分别调用 `NVPERF_EVENT_START()`、`NVPERF_EVENT_END()`进行测量；
4. 创建 `vector<double>`并调用 `NVPERF_EVENT_ELAPSED()`读取测量结果（单位$\mu{s}$）；
5. 编写数据后处理代码；
6. 调用 `NVPERF_EVENT_DESTROY()`销毁事件、释放资源。

以 `example/src/vadd.cu`为例：

```cpp
/* --------- device code --------- */
...
// 注册事件
NVPERF_EVENT(gadd);

__global__ void gadd_kernel(float *x, float *y, float *out, int n) {
    // 标记事件开始
    NVPERF_EVENT_START(gadd);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // for time measurement
    int repeat_cnt = 100000;
    if (tid < n) {
        while (repeat_cnt--)
        out[tid] = x[tid] + y[tid];
    }
    // 标记事件结束
    NVPERF_EVENT_END(gadd);
}
...

/* ---------- host code ---------- */ 
...
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    // 事件初始化
    NVPERF_EVENT_INIT(gadd, grid_size);
    gadd_kernel<<<grid_size, block_size>>>(cuda_x, cuda_y, cuda_out, N);    // one calculation per thread
    std::vector<double> timestamp(grid_size * 2);
    // 读取结果
    NVPERF_EVENT_ELAPSED(gadd, timestamp);
    // 数据后处理
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
...
    // 销毁事件
    NVPERF_EVENT_DESTROY(gadd);
```

在 `example`目录执行 `make vadd`进行编译（如果计算能力不是8.9需要自行查询，如7.5需指定 `make vadd SM_VER=75`），完成编译后运行 `./build/vadd` 便可以查看到gadd_kernel的执行耗时，但不同block的耗时可能存在较大差异，这可能是由其他任务如操作系统抢占GPU导致的（待考证），实际进行测试中应尽量保证GPU上没有其他后台程序。

### 循环计时

对于需要多次执行的代码片段如涉及多次调用的设备函数，需要采用循环计时的方式。循环计时会累加每次执行被测代码段的执行时间，但无法记录每次执行的时间戳。

### 主机计时

主机计时分为同步计时和异步计时。同步计时的时间与主机侧同步，主要用于测量主机代码的耗时，若需测量kernel时间，需要在测量片段中手动引入设备同步；而异步计时的时间戳来自设备侧，主要用于测量设备侧操作的耗时，不会引入设备同步，仅在统计耗时时会等待结束事件，若此时结束事件已完成则不会引入阻塞，反之会阻塞主机代码。

> ❗️异步计时可能会包含主机代码的执行耗时
>
> 异步计时基于CUDA Event设计，考虑这样一种情况，记录起始事件时GPU对应stream中没有其他任务，此时 `cudaEventRecord()`入队后就会立刻执行、记录GPU当前的时间戳，此时如果距离kernel调用还有很长一段主机代码，那么这段主机代码的耗时就会影响对kernel的计时，因此异步计时应尽可能靠近kernel调用位置。

## 准确性论证

TODO

## TODOs

* 添加更多example（List: prefix_sum, Top-K）、优化代码并完善文档，顺便练练CUDA😋。
* 对代码进行充分测试。
* 代码的健壮性并不是很好，比如同一个 `NVPERF_EVENT()`注册多次，报错是由编译器提供的，不是很直观。
  * 原来的想法是在宏函数中定义一个宏然后使用 `#ifdef`判断是否有重复调用并确保没有遗漏某些初始化，但宏函数中并不支持其他宏指令（因为 `#`本身有宏参数字符串化的含义）。
  * 但GoogleTest中重复定义测试集的报错也是直接由编译器提供的，可能确实没有更好且不太复杂的方案了？
* 构思可视化所有block执行历程的实现方案，并基于此设计实验探索GPGPU架构教科书中关于GPU任务warp调度、block调度的相关知识，完成nvperf方案准确性论证。
* 完成剩余任务的需求分析、可行性验证以及具体实现。

## 使用ncu源码级分析

<简要介绍ncu>

### 给可执行文件添加调试信息

```bash
# 添加调试信息
nvcc -lineinfo -g -o build/vadd vadd.cu

# ncu抓取运行时信息
ncu 
```

#ifndef RVTX_CUDA_BENCHMARK_CUH
#define RVTX_CUDA_BENCHMARK_CUH

#include "rvtx/core/benchmark.hpp"

namespace rvtx::cuda
{
    double timer_ms( const Benchmark::Task & task );
}

#endif // RVTX_CUDA_BENCHMARK_CUH
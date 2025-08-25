#ifndef RVTX_CUDA_MEMORY_CUH
#define RVTX_CUDA_MEMORY_CUH

#include <cuda_runtime_api.h>

#include "rvtx/cuda/setup.cuh"

namespace rvtx::cuda
{
    enum class MemcpyType
    {
        DeviceToHost,
        HostToDevice
    };

    template<MemcpyType type, class Type>
    void mmemcpy( Type * const dst, const Type * const src, std::size_t count )
    {
        const std::size_t size = count * sizeof( Type );
        if constexpr ( type == MemcpyType::HostToDevice )
        {
            cudaCheck( cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice ) );
        }
        else if constexpr ( type == MemcpyType::DeviceToHost )
        {
            cudaCheck( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost ) );
        }
    }

    template<class Type>
    void copy( Type * const __restrict__ dst, const Type * const __restrict__ src, uint32_t count );

    template<class Type1, class Type2>
    __host__ void prefixSumCount( Type1 * src, Type2 * dst, Type2 * countValue, uint32_t size );
} // namespace rvtx::cuda

#include "rvtx/cuda/memory.inl"

#endif // RVTX_CUDA_MEMORY_CUH

#include <thrust/device_ptr.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

#include "rvtx/cuda/memory.cuh"

namespace rvtx::cuda
{
#ifdef __CUDACC__
    template<class Type>
    __global__ void copyImpl( Type * const __restrict__ dst, const Type * const __restrict__ src, std::size_t count )
    {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x )
        {
            dst[ i ] = src[ i ];
        }
    }

    template<class Type>
    void copy( Type * const __restrict__ dst, const Type * const __restrict__ src, uint32_t count )
    {
        auto [ numBlocks, numThreads ] = KernelConfig::From( count, 256 );
        copyImpl<<<numBlocks, numThreads>>>( dst, src, count );
        cudaCheck( "Device to Device copy failed" );
    }

    template<class Type1, class Type2>
    __host__ void prefixSumCount( Type1 * src, Type2 * dst, Type2 * countValue, uint32_t size )
    {
        if constexpr ( std::is_same_v<Type1, Type2> )
        {
            thrust::exclusive_scan( thrust::device,
                                    thrust::device_ptr<Type1>( src ),
                                    thrust::device_ptr<Type1>( src + size ),
                                    thrust::device_ptr<Type1>( dst ) );
        }
        else
        {
            thrust::transform_exclusive_scan(
                thrust::device,
                thrust::device_ptr<Type1>( src ),
                thrust::device_ptr<Type1>( src + size ),
                thrust::device_ptr<Type2>( dst ),
                [] __device__( Type1 c ) { return static_cast<Type2>( c ); },
                0,
                thrust::plus<Type2>() );
        }

        mmemcpy<MemcpyType::DeviceToHost>( countValue, dst + size - 1, 1 );
    }
#endif // __CUDACC__
} // namespace rvtx::cuda

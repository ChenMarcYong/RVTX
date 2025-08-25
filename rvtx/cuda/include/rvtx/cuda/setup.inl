#include "rvtx/core/logger.hpp"
#include "rvtx/cuda/setup.cuh"

namespace rvtx::cuda
{
    inline void cudaCheck( std::string_view msg ) { cudaCheck( msg, cudaGetLastError() ); }

    inline void cudaCheck( std::string_view msg, cudaError_t err )
    {
        if ( err != cudaSuccess )
            rvtx::logger::error( "{}: {}", msg, cudaGetErrorString( err ) );
    }

    inline void cudaCheck( cudaError_t err )
    {
        if ( err != cudaSuccess )
            rvtx::logger::error( cudaGetErrorString( err ) );
    }

    inline KernelConfig KernelConfig::From( uint32_t n, uint32_t blockDim )
    {
        blockDim = std::min( blockDim, n );
        return { n / blockDim + std::min( n / blockDim, 1u ), blockDim };
    }

    inline KernelConfig::operator std::tuple<dim3, dim3>() const { return { gridDim, blockDim }; }
} // namespace rvtx::cuda
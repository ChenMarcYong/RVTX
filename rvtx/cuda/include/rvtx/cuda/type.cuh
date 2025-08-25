#ifndef RVTX_CUDA_TYPE_HPP
#define RVTX_CUDA_TYPE_HPP

#include <cstddef>

namespace rvtx::cuda
{
    template<class Type>
    __host__ __device__ std::size_t offsetSize( std::size_t nb );
}

#include "rvtx/cuda/type.inl"

#endif // RVTX_CUDA_TYPE_HPP

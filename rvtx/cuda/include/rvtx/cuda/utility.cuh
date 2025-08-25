#ifndef RVTX_CUDA_UTILITY_HPP
#define RVTX_CUDA_UTILITY_HPP

namespace rvtx::cuda
{
    template<class Type>
    __device__ __host__ void swap( Type & a, Type & b ) noexcept;
}

#include "rvtx/cuda/utility.inl"

#endif // RVTX_CUDA_UTILITY_HPP

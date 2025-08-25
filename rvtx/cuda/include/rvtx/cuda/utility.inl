#include "rvtx/cuda/utility.cuh"

namespace rvtx::cuda
{
    template<class Type>
    __device__ __host__ void swap( Type & a, Type & b ) noexcept
    {
        const Type temp = a;
        a               = b;
        b               = temp;
    }
} // namespace rvtx::cuda

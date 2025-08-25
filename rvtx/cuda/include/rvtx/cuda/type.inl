#include "rvtx/cuda/type.cuh"

namespace rvtx::cuda
{
    template<class Type>
    __host__ __device__ std::size_t offsetSize( std::size_t nb )
    {
        return sizeof( Type ) * nb;
    }

} // namespace rvtx::cuda

#ifndef RVTX_CUDA_EXECUTION_CUH
#define RVTX_CUDA_EXECUTION_CUH

namespace rvtx::cuda
{
    template<uint32_t WarpThreadNb = 32>
    __device__ uint32_t getIdFromMask( uint32_t mask )
    {
        return ( WarpThreadNb - 1u ) - __clz( mask );
    }

    // Use the mask as a processing queue. Every 1 in the mask's bit representation
    // will produce a call to predicate with its corresponding index in the mask.
    template<uint32_t WarpThreadNb, typename Predicate>
    __device__ void executeQueue( uint32_t mask, Predicate predicate )
    {
        while ( mask )
        {
            const uint32_t id = getIdFromMask( mask );
            predicate( id );
            mask &= ~( 1u << id );
        }
    }
} // namespace rvtx::cuda

#endif // RVTX_CUDA_EXECUTION_CUH

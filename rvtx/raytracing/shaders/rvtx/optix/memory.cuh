#ifndef RVTX_SHADERS_OPTIX_MEMORY_CUH
#define RVTX_SHADERS_OPTIX_MEMORY_CUH

namespace rvtx::optix
{
    template<class Type>
    __host__ __device__ uint2 u64ToU32x2( Type * ptr )
    {
        const uint64_t uPtr = reinterpret_cast<uint64_t>( ptr );
        return make_uint2( uPtr >> 32, uPtr & 0x00000000ffffffff );
    }

    template<class Type>
    __host__ __device__ Type * u32x2ToType( uint2 packed )
    {
        return reinterpret_cast<Type *>( static_cast<uint64_t>( packed.x ) << 32 | packed.y );
    }
} // namespace rvtx::optix

#endif // RVTX_SHADERS_OPTIX_MEMORY_CUH
#ifndef RVTX_MATERIAL_FRESNEL_CUH
#define RVTX_MATERIAL_FRESNEL_CUH

namespace rvtx::optix
{
    __device__ inline float  iorToReflectance( float ior );
    __device__ inline float3 fresnelSchlick( float3 r0, float u );
} // namespace rvtx::optix

#include "rvtx/optix/material/fresnel.inl"

#endif // RVTX_MATERIAL_FRESNEL_CUH
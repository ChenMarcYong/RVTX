#ifndef RVTX_OPTIX_MATERIAL_LAMBERTIAN_CUH
#define RVTX_OPTIX_MATERIAL_LAMBERTIAN_CUH

#include "rvtx/cuda/random.cuh"

namespace rvtx::optix
{
    inline __device__ float3 sampleLambertian( float3 wo, float2 u );
    inline __device__ float3 evalLambertian( float3 baseColor );
    inline __device__ float  getPdfLambertian( float3 wo, float3 wi );
} // namespace rvtx::optix

#include "rvtx/optix/material/lambertian.inl"

#endif // RVTX_OPTIX_MATERIAL_LAMBERTIAN_CUH

#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/material/lambertian.cuh"

namespace rvtx::optix
{
    inline __device__ float3 sampleLambertian( const float3 wo, const float2 u )
    {
        float3 wi = normalize( cuda::randomInCosineWeightedHemisphere( u ) );
        if ( wo.z < 0.f )
            wi.z *= -1.f;
        return wi;
    }

    inline __device__ float3 evalLambertian( float3 baseColor ) { return baseColor * rvtx::OneOverPi; }

    inline __device__ float getPdfLambertian( const float3 wo, const float3 wi )
    {
        return wo.z * wi.z > 0.f ? fabs( wi.z ) / rvtx::Pi : 0.f;
    }

} // namespace rvtx::optix

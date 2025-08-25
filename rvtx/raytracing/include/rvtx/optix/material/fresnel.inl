#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/material/fresnel.cuh"

namespace rvtx::optix
{
    __device__ inline float iorToReflectance( float ior )
    {
        const float a = ior - 1.f;
        const float b = ior + 1.f;
        return ( a * a ) / ( b * b );
    }

    // Schlick, C. (1994). An Inexpensive BRDF Model for Physically-based Rendering.
    // In Computer Graphics Forum (Vol. 13, Issue 3, pp. 233–246). Wiley.
    __device__ inline float3 fresnelSchlick( float3 r0, float u ) { return r0 + ( 1. - r0 ) * powf( 1.f - u, 5.f ); }
} // namespace rvtx::optix

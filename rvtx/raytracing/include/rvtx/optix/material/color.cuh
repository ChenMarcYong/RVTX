#ifndef RVTX_OPTIX_MATERIAL_COLOR_CUH
#define RVTX_OPTIX_MATERIAL_COLOR_CUH

namespace rvtx::optix
{
    __device__ __host__ float3 linearToSRGB( const float3 color, const float gamma = 1.f / 2.2f );
    __device__ __host__ float4 linearToSRGB( const float4 color, const float gamma = 1.f / 2.2f );
    __device__ __host__ uchar4 float3ToColor( float3 color );
    __device__ __host__ uchar4 float4ToColor( float4 color );

    __host__ __device__ float  getLuminance( const float3 rgb );
    __device__ float3 ACESFilm( const float3 x );
    __device__ float3 fadedBackground( const float3 rd );
} // namespace rvtx::optix

#include "rvtx/optix/material/color.inl"

#endif // RVTX_OPTIX_MATERIAL_COLOR_CUH

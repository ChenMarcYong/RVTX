#ifndef RVTX_OPTIX_ENVIRONMENT_CUH
#define RVTX_OPTIX_ENVIRONMENT_CUH

#include <vector>

#include "rvtx/core/type.hpp"
#include "rvtx/optix/texture.cuh"

namespace rvtx::optix
{
    std::vector<float> getSamplingImage( uint32_t               width,
                                         uint32_t               height,
                                         uint32_t               channels,
                                         rvtx::ConstSpan<float> data );

    __device__ inline float4 getEnvironmentColor( const Texture & environment, float3 rd, float lod = 0.f );
    __device__ inline float4 sample2D( const Texture & environment, float2 u, float & pdf );
    __device__ inline float4 sampleEnvironment( const Texture & environment, float2 u, float & pdf );
    __device__ inline float  getPdfEnvironment( const Texture::View & environment, float3 v );
} // namespace rvtx::optix

#endif // RVTX_OPTIX_ENVIRONMENT_CUH
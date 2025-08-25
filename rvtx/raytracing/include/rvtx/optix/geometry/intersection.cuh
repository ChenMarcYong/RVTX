#ifndef RVTX_OPTIX_GEOMETRY_CUH
#define RVTX_OPTIX_GEOMETRY_CUH

namespace rvtx::optix
{
    __device__ float2 iBox( float3 ro, float3 rd, float3 pMin, float3 pMax );
    __device__ float2 iOOBB( float3 ro, float3 rd, float3 p, float3 dim, float4 rot );
    __device__ float2 iSphere( float3 ro, float3 rd, float4 sph );
} // namespace rvtx::optix

#include "rvtx/optix/geometry/intersection.inl"

#endif // RVTX_OPTIX_GEOMETRY_HPP

#ifndef RVTX_OPTIX_MATERIAL_TROWBRIDGEREITZGGX_CUH
#define RVTX_OPTIX_MATERIAL_TROWBRIDGEREITZGGX_CUH

namespace rvtx::optix
{
    inline __device__ float  getSmithG1GGX( float sn2, float alpha2 );
    inline __device__ float  getSmithG2GGX( float won, float win, float alpha2 );
    inline __device__ float  getDGGX( float hn, float alpha2 );
    inline __device__ float3 sampleGGXVNDF( float3 V_, float alpha_x, float alpha_y, float U1, float U2 );
} // namespace rvtx::optix

#include "rvtx/optix/material/trowbridge_reitz_ggx.inl"

#endif // RVTX_OPTIX_MATERIAL_TROWBRIDGEREITZGGX_CUH

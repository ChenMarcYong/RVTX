#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_CUH
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_CUH

#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/setup.cuh"

namespace rvtx::optix
{
    void fillSpheres( const uint32_t sphereNb, const float4 * spheres, OptixAabb * aabbs );
}

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_CUH
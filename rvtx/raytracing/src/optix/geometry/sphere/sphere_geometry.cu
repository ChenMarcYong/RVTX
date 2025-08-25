#include <device_launch_parameters.h>

#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/geometry/sphere/sphere_geometry.cuh"

namespace rvtx::optix
{
    __global__ void doFillSpheres( const uint32_t sphereNb, const float4 * spheres, OptixAabb * aabbs );

    void fillSpheres( const uint32_t sphereNb, const float4 * spheres, OptixAabb * aabbs )
    {
        auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( sphereNb, 256 );
        doFillSpheres<<<numBlocks, numThreads>>>( sphereNb, spheres, aabbs );
    }

    __global__ void doFillSpheres( const uint32_t sphereNb, const float4 * spheres, OptixAabb * aabbs )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= sphereNb )
            return;

        const float4 sphere   = spheres[ id ];
        const float3 position = make_float3( sphere );

        const float3 max = position + make_float3( sphere.w );
        const float3 min = position - make_float3( sphere.w );

        OptixAabb aabb;
        aabb.minX   = min.x;
        aabb.minY   = min.y;
        aabb.minZ   = min.z;
        aabb.maxX   = max.x;
        aabb.maxY   = max.y;
        aabb.maxZ   = max.z;
        aabbs[ id ] = aabb;
    }
} // namespace rvtx::optix

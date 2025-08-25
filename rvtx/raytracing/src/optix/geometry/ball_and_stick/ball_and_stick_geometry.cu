#include <device_launch_parameters.h>

#include "rvtx/cuda/math.cuh"
#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_data.hpp"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.cuh"
#include "rvtx/optix/setup.cuh"

namespace rvtx::optix
{
    __global__ void fillBAndSSpheres( uint32_t    sphereNb,
                                      float4 *    spheres,
                                      float       radius,
                                      OptixAabb * aabbs,
                                      uint32_t *  sbtIndex );
    __global__ void fillBAndSBonds( uint32_t       bondNb,
                                    const float4 * spheres,
                                    const uint2 *  bonds,
                                    float          bondRadius,
                                    OptixAabb *    aabbs,
                                    uint32_t *     sbtIndex );

    void fillBallAndStick( uint32_t      sphereNb,
                           float4 *      spheres,
                           float         sphereRadius,
                           uint32_t      bondNb,
                           const uint2 * bonds,
                           const float   bondRadius,
                           OptixAabb *   aabbs,
                           uint32_t *    sbtIndex )
    {
        if ( sphereNb > 0 )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( sphereNb, 256 );
            fillBAndSSpheres<<<numBlocks, numThreads>>>( sphereNb, spheres, sphereRadius, aabbs, sbtIndex );
        }
        if ( bondNb > 0 )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( bondNb, 256 );
            fillBAndSBonds<<<numBlocks, numThreads>>>(
                bondNb, spheres, bonds, bondRadius, aabbs + sphereNb, sbtIndex + sphereNb );
        }
    }

    __global__ void fillBAndSSpheres( uint32_t    sphereNb,
                                      float4 *    spheres,
                                      float       sphereRadius,
                                      OptixAabb * aabbs,
                                      uint32_t *  sbtIndex )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= sphereNb )
            return;

        sbtIndex[ id ] = BallAndStickHitGroupData::SbtIndexSphere;

        spheres[ id ].w       = sphereRadius;
        const float4 sphere   = spheres[ id ];
        const float3 position = make_float3( sphere );

        const float3 min = position - make_float3( sphere.w );
        const float3 max = position + make_float3( sphere.w );

        OptixAabb aabb;
        aabb.minX   = min.x;
        aabb.minY   = min.y;
        aabb.minZ   = min.z;
        aabb.maxX   = max.x;
        aabb.maxY   = max.y;
        aabb.maxZ   = max.z;
        aabbs[ id ] = aabb;
    }

    __global__ void fillBAndSBonds( uint32_t       bondNb,
                                    const float4 * spheres,
                                    const uint2 *  bonds,
                                    const float    bondRadius,
                                    OptixAabb *    aabbs,
                                    uint32_t *     sbtIndex )
    {
        const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if ( id >= bondNb )
            return;

        sbtIndex[ id ] = BallAndStickHitGroupData::SbtIndexBond;

        const uint2  bond  = bonds[ id ];
        const float3 start = make_float3( spheres[ bond.x ] );
        const float3 end   = make_float3( spheres[ bond.y ] );

        // The MIT License
        // Copyright © 2016 Inigo Quilez
        // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
        // documentation files (the "Software"), to deal in the Software without restriction, including without
        // limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
        // Software, and to permit persons to whom the Software is furnished to do so, subject to the following
        // conditions: The above copyright notice and this permission notice shall be included in all copies or
        // substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        // OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
        // AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
        // OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
        // WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        const float3 a   = end - start;
        const float3 e   = bondRadius * cuda::sqrtf( 1.f - a * a / dot( a, a ) );
        const float3 min = fminf( start - e, end - e );
        const float3 max = fmaxf( start + e, end + e );

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

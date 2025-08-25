#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_CUH
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_CUH

#include "rvtx/cuda/setup.cuh"

struct OptixAabb;

namespace rvtx::optix
{
    void fillBallAndStick( uint32_t      sphereNb,
                           float4 *      spheres,
                           float         sphereRadius,
                           uint32_t      bondNb,
                           const uint2 * bonds,
                           const float   bondRadius,
                           OptixAabb *   aabbs,
                           uint32_t *    sbtIndex );
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_CUH

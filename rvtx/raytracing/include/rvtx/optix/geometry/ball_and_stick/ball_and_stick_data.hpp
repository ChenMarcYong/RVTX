#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_DATA_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_DATA_HPP

namespace rvtx::optix
{
    struct BallAndStickHitGroupData
    {
        float4 * spheres; // Must be the first pointer
        uint32_t sphereNb;
        uint2 *  bonds;
        uint32_t bondNb;
        float    bondRadius;

        constexpr static uint32_t SbtIndexSphere = 0;
        constexpr static uint32_t SbtIndexBond   = 1;
    };
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_DATA_HPP

#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_DATA_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_DATA_HPP

#include <bcs/sesdf/graphics.hpp>

namespace rvtx::optix
{
    struct CircleBoundingGeometry
    {
        float4 bbPos;
        float4 bbDim;
        float4 rotation;
        float4 vSphere;
    };

    struct SesdfHitGroupData
    {
        constexpr static uint32_t SbtIndexConvex  = 0;
        constexpr static uint32_t SbtIndexConcave = 1;
        constexpr static uint32_t SbtIndexCircle  = 2;
        constexpr static uint32_t SbtIndexSegment = 3;

        CircleBoundingGeometry * circlesBb;
        CircleBoundingGeometry * segmentsBb;

        bcs::sesdf::SesdfData data;
    };
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_DATA_HPP

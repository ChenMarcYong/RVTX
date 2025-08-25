#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_CUH
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_CUH

#include <bcs/sesdf/graphics.hpp>
#include <optix_types.h>

#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/geometry/ses/ses_data.hpp"

namespace rvtx::optix
{
    __host__ void   fillPatches( bcs::sesdf::SesdfData    data,
                                 OptixAabb *              result,
                                 uint32_t *               sbtIndex,
                                 CircleBoundingGeometry * circlesBb,
                                 CircleBoundingGeometry * segmentsBb );
    __global__ void fillConvexPatches( bcs::sesdf::SesdfData data, OptixAabb * result, uint32_t * sbtIndex );
    __global__ void fillConcavePatches( bcs::sesdf::SesdfData data, OptixAabb * result, uint32_t * sbtIndex );
    __global__ void fillCirclePatches( bcs::sesdf::SesdfData    data,
                                       OptixAabb *              result,
                                       uint32_t *               sbtIndex,
                                       CircleBoundingGeometry * circlesBb );
    __global__ void fillSegmentPatches( bcs::sesdf::SesdfData    data,
                                        OptixAabb *              result,
                                        uint32_t *               sbtIndex,
                                        CircleBoundingGeometry * segmentsBb );
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_CUH
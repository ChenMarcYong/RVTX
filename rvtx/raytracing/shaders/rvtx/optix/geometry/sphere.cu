#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/optix/geometry/intersection.cuh>

#include "rvtx/optix/data.cuh"
#include "rvtx/optix/memory.cuh"

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

extern "C" __global__ void __closesthit__sphere()
{
    const unsigned int     u0      = optixGetPayload_0();
    const unsigned int     u1      = optixGetPayload_1();
    rvtx::optix::HitInfo * hitInfo = rvtx::optix::u32x2ToType<rvtx::optix::HitInfo>( make_uint2( u0, u1 ) );

    const auto * hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto * spheres  = reinterpret_cast<const float4 *>( hitGroup->userPtr );

    const uint32_t id     = optixGetPrimitiveIndex();
    const float4   sphere = spheres[ id ];

    hitInfo->position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

    float3 objectNormal   = ( hitInfo->position - make_float3( sphere ) ) / sphere.w;
    hitInfo->objectNormal = objectNormal;
    hitInfo->worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );
    hitInfo->t            = optixGetRayTmax();
    hitInfo->hit          = true;

    const auto * materials = hitGroup->materials;
    hitInfo->material      = materials[ id ];
}

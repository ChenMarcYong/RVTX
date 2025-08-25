#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/optix/geometry/ses/ses_data.hpp>

#include "rvtx/optix/data.cuh"
#include "rvtx/optix/memory.cuh"

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

__device__ bool isInSector( const float3 & p, const float3 & o, float r )
{
    // Reference: https://www.shadertoy.com/view/wsyyRh
    return acos( clamp( dot( p, o ), -1.f, 1.f ) ) - r < 0.f;
}

__device__ bool isInPatch( const float3 & p,
                           const float3 & center,
                           const float    radius,
                           const uint2 &  ids,
                           const float4 * const __restrict__ sectors )
{
    const float3 cp = ( p - center ) / radius;

    bool isIn = true;
    for ( uint32_t i = ids.x; isIn && i < ids.y && isIn; i++ )
    {
        const float4 sector = sectors[ i ];
        isIn                = !isInSector( cp, make_float3( sector ), sector.w );
    }

    return isIn;
}

extern "C" __global__ void __intersection__convex()
{
    const auto *   hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   sesdf    = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const uint32_t id       = optixGetPrimitiveIndex();

    if ( id >= sesdf->data.atomNb )
        return;

    const float4 * atoms   = sesdf->data.atoms;
    const uint2    indices = sesdf->data.convexPatches[ id ];

    const float4 * sectors = sesdf->data.sectors;
    const float4   atom    = atoms[ id ];
    const float3   atomPos = make_float3( atom );

    // Improving Numerical Precision in Intersection Programs, Ingo Walds, Raytracing Gems 2
    float3       ro           = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd           = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );
    const float  distToCenter = length( ro - atomPos );
    float        toAdd        = 0.f;
    if ( distToCenter > atom.w )
        toAdd = ( distToCenter - atom.w );
    ro += rd * toAdd;

    const float3 oc = ro - atomPos;
    const float  b  = dot( oc, rd );
    const float  c  = dot( oc, oc ) - atom.w * atom.w;
    const float  h  = b * b - c;

    if ( h > 0.f )
    {
        const float tMin = optixGetRayTmin();
        const float tMax = optixGetRayTmax();

        float3 hit;
        float  dist   = -b - sqrtf( h );
        bool   hitSes = dist + toAdd > tMin && isInPatch( hit = ro + rd * dist, atomPos, atom.w, indices, sectors );

        if ( !hitSes && ( dist = -b + sqrtf( h ) ) + toAdd > tMin
             && isInPatch( hit = ro + rd * dist, atomPos, atom.w, indices, sectors ) )
            hitSes = true;

        dist += toAdd;
        if ( hitSes && dist > tMin && dist < tMax )
        {
            const float3 normal = normalize( hit - atomPos );
            optixReportIntersection( dist, 0, float3_as_ints( normal ), __float_as_int( atom.w ) );
        }
    }
}

extern "C" __global__ void __closesthit__convex()
{
    const unsigned int     u0      = optixGetPayload_0();
    const unsigned int     u1      = optixGetPayload_1();
    rvtx::optix::HitInfo * hitInfo = rvtx::optix::u32x2ToType<rvtx::optix::HitInfo>( make_uint2( u0, u1 ) );

    float3 objectNormal   = make_float3( __int_as_float( optixGetAttribute_0() ),
                                       __int_as_float( optixGetAttribute_1() ),
                                       __int_as_float( optixGetAttribute_2() ) );
    hitInfo->objectNormal = objectNormal;
    hitInfo->worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );
    hitInfo->t            = optixGetRayTmax();
    hitInfo->position     = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    hitInfo->hit          = true;

    const auto *   hitGroup  = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   materials = hitGroup->materials;
    const uint32_t id        = optixGetPrimitiveIndex();
    hitInfo->material        = materials[ id ];
}

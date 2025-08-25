#include <bcs/cuda/circles.cuh>
#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/optix/geometry/intersection.cuh>
#include <rvtx/optix/geometry/ses/ses_data.hpp>

#include "rvtx/optix/data.cuh"
#include "rvtx/optix/memory.cuh"

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

inline __host__ __device__ float3 orthogonalVector( const float3 & normal )
{
    float3 ref = { 1.f, 0.f, 0.f };
    if ( fabsf( dot( normal, ref ) ) > 1.f - 1e-4f )
        ref = make_float3( 0.f, 1.f, 0.f );
    return normalize( cross( normal, ref ) );
}

__device__ float3 closestPointOnCircle( const float3 circleCenter,
                                        const float3 circleNormal,
                                        float        circleRadius,
                                        const float3 point )
{
    const float3 p1 = point + dot( circleCenter - point, circleNormal ) * circleNormal;

    const float3 circleCenterToP1 = p1 - circleCenter;
    const float  distToP1         = length( circleCenterToP1 );
    if ( abs( distToP1 ) > 0.f )
    {
        return circleCenter + circleRadius * circleCenterToP1 / distToP1;
    }
    else
    {
        const float3 v = orthogonalVector( circleNormal );
        return circleCenter + circleRadius * v;
    }
}

__device__ float sdToroidalPatch( const float3 point,
                                  const float3 circleCenter,
                                  const float3 circleNormal,
                                  const float  circleRadius,
                                  const float  probeRadius,
                                  float3 &     closestPoint )
{
    closestPoint = closestPointOnCircle( circleCenter, circleNormal, circleRadius, point );
    return -length( point - closestPoint ) + probeRadius;
}

extern "C" __global__ void __intersection__circle()
{
    const auto * hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto * sesdf    = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );

    const uint32_t offset = sesdf->data.convexPatchNb + sesdf->data.concavePatchNb;
    const uint32_t id     = optixGetPrimitiveIndex() - offset;

    if ( id > sesdf->data.circlePatchNb )
        return;

    rvtx::optix::CircleBoundingGeometry circleBb = sesdf->circlesBb[ id ];
    uint2                               atomIds  = sesdf->data.circlePatches[ id ];

    const float4 * atoms = sesdf->data.atoms;
    const float4   s     = atoms[ atomIds.x ];
    const float3   sp    = make_float3( s );

    const float4 e  = atoms[ atomIds.y ];
    const float3 ep = make_float3( e );

    const float pr  = sesdf->data.probeRadius;
    const float ier = s.w + pr;

    const float3 c = bcs::getCircleCenter( sp, ier, ep, e.w + pr );

    const float3 ci   = sp - c;
    const float  scd2 = dot( ci, ci );
    const float  r    = sqrtf( ier * ier - scd2 );

    float3       ro = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );

    // Improving Numerical Precision in Intersection Programs, Ingo Walds, Raytracing Gems 2
    const float dv    = length( ro - make_float3( circleBb.vSphere ) );
    float       toAdd = 0.f;
    if ( dv > circleBb.vSphere.w )
        toAdd = ( dv - circleBb.vSphere.w );
    ro += rd * toAdd;

    const float3 n = normalize( ep - sp );

    float2 bounds
        = rvtx::optix::iOOBB( ro, rd, make_float3( circleBb.bbPos ), make_float3( circleBb.bbDim ), circleBb.rotation );
    const float2 sbounds = rvtx::optix::iSphere( ro, rd, circleBb.vSphere );

    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

// #define CIRCLE_RENDER_BOUDING_SHAPE
#ifdef CIRCLE_RENDER_BOUDING_SHAPE
    if ( bounds.x != -1.f && sbounds.x != -1.f && ( bounds.y > tMin || sbounds.y > tMin ) )
        optixReportIntersection( toAdd + bounds.x, 0, float3_as_ints( -rd ), optixGetSbtGASIndex() );
    return;
#endif // CIRCLE_RENDER_BOUDING_SHAPE

    constexpr int   MaxSteps  = 75;
    constexpr float BaseBound = 2e-3f;
    if ( bounds.x != -1.f && sbounds.x != -1.f && ( bounds.y > tMin || sbounds.y > tMin ) )
    {
        bounds.x = fmax( bounds.x, fmax( sbounds.x, tMin ) );
        bounds.y = fmin( bounds.y, fmin( tMax, sbounds.y ) );

        float3 cl;
        int    i = 0;
        float  t = bounds.x;
        while ( t < bounds.y && i++ < MaxSteps )
        {
            const float3 p = ro + rd * t;
            const float  d = abs( sdToroidalPatch( p, c, n, r, pr, cl ) );
            if ( d < BaseBound )
            {
                const float3 normal = normalize( cl - p );
                const float  trueT  = toAdd + t;
                if ( tMin < trueT && tMax > trueT )
                    optixReportIntersection( trueT, 0, float3_as_ints( normal ), 0.f );
            }

            t += fmaxf( d, 1e-4f );
        }
    }
}

extern "C" __global__ void __closesthit__circle()
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

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    hitInfo->position     = position;
    hitInfo->hit          = true;

    const auto * hitGroup  = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto * sesdf     = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const auto * materials = hitGroup->materials;

    const uint32_t offset = sesdf->data.convexPatchNb + sesdf->data.concavePatchNb;
    const uint32_t id     = optixGetPrimitiveIndex() - offset;

    uint2          atomIds = sesdf->data.circlePatches[ id ];
    const float4 * atoms   = sesdf->data.atoms;
    const float3   start   = make_float3( atoms[ atomIds.x ] );
    const float3   end     = make_float3( atoms[ atomIds.y ] );

    const float d1 = length( start - position ) - atoms[ atomIds.x ].w;
    const float d2 = length( end - position ) - atoms[ atomIds.y ].w;
    if ( d1 < d2 )
    {
        hitInfo->material = materials[ atomIds.x ];
    }
    else
    {
        hitInfo->material = materials[ atomIds.y ];
    }
    float d                     = d1 + d2;
    hitInfo->material.baseColor = ( 1.f - ( d1 / d ) ) * materials[ atomIds.x ].baseColor
                                  + ( 1.f - ( d2 / d ) ) * materials[ atomIds.y ].baseColor;
}

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

__device__ float3 closestPointOnCircle( const float3 & circleCenter,
                                        const float3 & circleNormal,
                                        const float    circleRadius,
                                        const float3 & point )
{
    const float3 p1 = point + dot( circleCenter - point, circleNormal ) * circleNormal;

    const float3 circleCenterToP1 = p1 - circleCenter;
    const float  distToP1         = length( circleCenterToP1 );
    if ( abs( distToP1 ) > 1e-4f )
    {
        return circleCenter + circleRadius * circleCenterToP1 / distToP1;
    }
    else // If p1 == circle center
    {
        const float3 v = orthogonalVector( circleNormal );
        return circleCenter + circleRadius * v;
    }
}

__device__ float sdCircle( const float3 & p, const float3 & c, const float3 & n, float r )
{
    return length( closestPointOnCircle( c, n, r, p ) - p );
}

__device__ float sdToroidalPatch( const float3 & p,
                                  const float3 & c,
                                  const float3 & n,
                                  const float    r,
                                  const float3 & v1,
                                  const float3 & x1,
                                  const float3 & x1n,
                                  const float3 & v2,
                                  const float3 & x2,
                                  const float3 & x2n,
                                  const float    probeRadius,
                                  const float    maxAngle,
                                  float3 &       cl )
{
    cl = closestPointOnCircle( c, n, r, p );

    const float3 v  = ( cl - c ) / r;
    const float  a1 = bcs::angleBetweenEdges( v1, v, n );
    if ( a1 < maxAngle )
        return -length( p - cl ) + probeRadius;

    if ( length( p - x1 ) < length( p - x2 ) )
        return sdCircle( p, x1, x1n, probeRadius );
    return sdCircle( p, x2, x2n, probeRadius );
}

// #define SEGMENTS_RENDER_BOUDING_SHAPE
extern "C" __global__ void __intersection__segment()
{
    const auto *   hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   sesdf    = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const uint32_t offset   = sesdf->data.convexPatchNb + sesdf->data.concavePatchNb + sesdf->data.circlePatchNb;
    const uint32_t id       = optixGetPrimitiveIndex() - offset;

    if ( id > sesdf->data.segmentPatchNb )
        return;

    rvtx::optix::CircleBoundingGeometry segmentBb = sesdf->segmentsBb[ id ];
    const uint4                         data      = sesdf->data.segmentPatches[ id ];

    const uint32_t i = data.x;
    const uint32_t j = data.y;
    const uint32_t s = data.z;
    const uint32_t e = data.w;

    const float    maxAngle = segmentBb.bbPos.w;
    const float    pr       = sesdf->data.probeRadius;
    const float4 * atoms    = sesdf->data.atoms;
    const float4 * probes   = sesdf->data.concavePatchesPosition;

    const float3 ro = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );

    const float4 sA  = atoms[ i ];
    const float3 sAp = make_float3( sA );

    const float4 eA  = atoms[ j ];
    const float3 eAp = make_float3( eA );

    const float3 x1 = make_float3( probes[ s ] );
    const float3 x2 = make_float3( probes[ e ] );

    const float3 c = bcs::getCircleCenter( sAp, sA.w + pr, eAp, eA.w + pr );

    const float3 n = normalize( eAp - sAp );

    float3      v1 = x1 - c;
    const float r  = length( v1 );
    v1 /= r;
    const float3 v2 = ( x2 - c ) / r;

    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    const float  distToVSphere = length( ro - make_float3( segmentBb.vSphere ) ) - segmentBb.vSphere.w;
    const float  toAdd         = fmaxf( distToVSphere - tMin, 0.f );
    const float3 cro           = ro + rd * toAdd;

    float2 bounds = rvtx::optix::iOOBB(
        cro, rd, make_float3( segmentBb.bbPos ), make_float3( segmentBb.bbDim ), segmentBb.rotation );
    const float2 sbounds = rvtx::optix::iSphere( cro, rd, segmentBb.vSphere );

// #define SEGMENTS_RENDER_BOUDING_SHAPEs
#ifdef SEGMENTS_RENDER_BOUDING_SHAPE
    if ( bounds.x != -1.f && sbounds.x != -1.f && ( bounds.y > tMin || sbounds.y > tMin ) )
        optixReportIntersection( toAdd + bounds.x, 0, float3_as_ints( -rd ), optixGetSbtGASIndex() );
    return;
#endif // SEGMENTS_RENDER_BOUDING_SHAPE

    constexpr int   MaxSteps  = 150;
    constexpr float BaseBound = 1e-3f;
    if ( bounds.x != -1.f && sbounds.x != -1.f && ( bounds.y > tMin || sbounds.y > tMin ) )
    {
        bounds.x = fmax( bounds.x, fmax( sbounds.x, tMin ) );
        bounds.y = fmin( bounds.y, fmin( tMax, sbounds.y ) );

        const float3 x1n = normalize( cross( n, v1 ) );
        const float3 x2n = normalize( cross( n, v2 ) );

        float3 cl;
        int    i = 0;
        float  t = bounds.x;
        while ( t < bounds.y && i++ < MaxSteps )
        {
            const float3 p = cro + rd * t;
            const float  d = abs( sdToroidalPatch( p, c, n, r, v1, x1, x1n, v2, x2, x2n, pr, maxAngle, cl ) );
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

extern "C" __global__ void __closesthit__segment()
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

    const auto *   hitGroup  = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   sesdf     = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const auto *   materials = hitGroup->materials;
    const uint32_t offset    = sesdf->data.convexPatchNb + sesdf->data.concavePatchNb + sesdf->data.circlePatchNb;
    const uint32_t id        = optixGetPrimitiveIndex() - offset;
    const uint4    data      = sesdf->data.segmentPatches[ id ];

    const uint32_t i = data.x;
    const uint32_t j = data.y;
    const uint32_t s = data.z;
    const uint32_t e = data.w;

    const float    pr     = sesdf->data.probeRadius;
    const float4 * atoms  = sesdf->data.atoms;
    const float4 * probes = sesdf->data.concavePatchesPosition;

    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();

    const float4 sA  = atoms[ i ];
    const float3 sAp = make_float3( sA );

    const float4 eA  = atoms[ j ];
    const float3 eAp = make_float3( eA );

    const float3 x1 = make_float3( probes[ s ] );
    const float3 x2 = make_float3( probes[ e ] );

    const float3 c = bcs::getCircleCenter( sAp, sA.w + pr, eAp, eA.w + pr );
    const float3 n = normalize( eAp - sAp );

    float3      v1 = x1 - c;
    const float r  = length( v1 );

    const float3 p           = c + orthogonalVector( n ) * r;
    const float4 sCircle     = bcs::getSmallCircle( sA, n, p );
    const float3 sCirclePos  = make_float3( sCircle );
    const float4 sCircle2    = bcs::getSmallCircle( eA, n, p );
    const float3 sCircle2Pos = make_float3( sCircle2 );

    float d1 = length( closestPointOnCircle( sCirclePos, n, sCircle.w, position ) - position );
    float d2 = length( closestPointOnCircle( sCircle2Pos, n, sCircle2.w, position ) - position );
    if ( d1 < d2 )
    {
        hitInfo->material = materials[ data.x ];
    }
    else
    {
        hitInfo->material = materials[ data.y ];
    }

    const float w1 = d1;
    const float w2 = d2;
    const float w  = w1 + w2;

    hitInfo->material.baseColor
        = ( w - w1 ) * materials[ data.x ].baseColor / w + ( w - w2 ) * materials[ data.y ].baseColor / w;
}

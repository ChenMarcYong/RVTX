#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/optix/geometry/ses/ses_data.hpp>

#include "rvtx/optix/data.cuh"
#include "rvtx/optix/memory.cuh"

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

__device__ float sdPlane( const float3 & pos, const float3 & n, const float d ) { return dot( pos, n ) - d; }

__device__ bool isValid( const float3 & point,
                         const uint32_t startNeighborId,
                         const uint32_t neighborNb,
                         const float    probeRadius,
                         const float4 * probesNeighbors )
{
    uint32_t baseId = startNeighborId;
    uint32_t maxId  = baseId + neighborNb;

    bool valid = true;
    for ( uint32_t otherId = baseId; valid && otherId < maxId; otherId++ )
    {
        const float3 otherIntersection = make_float3( probesNeighbors[ otherId ] );
        valid                          = length( point - otherIntersection ) - probeRadius > -1e-3f;
    }

    return valid;
}

__device__ bool isInConcavePatch( const float3 & point,
                                  const uint32_t startNeighborId,
                                  const uint32_t neighborNb,
                                  const float3 & p1n,
                                  const float    p1d,
                                  const float3 & p2n,
                                  const float    p2d,
                                  const float3 & p3n,
                                  const float    p3d,
                                  const float    probeRadius,
                                  const float4 * probeNeighbors )
{
    return sdPlane( point, p1n, p1d ) > 1e-4f && sdPlane( point, p2n, p2d ) > 1e-4f
           && sdPlane( point, p3n, p3d ) > 1e-4f
           && isValid( point, startNeighborId, neighborNb, probeRadius, probeNeighbors );
}

extern "C" __global__ void __intersection__concave()
{
    const auto *   hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   sesdf    = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const uint32_t id       = optixGetPrimitiveIndex() - sesdf->data.convexPatchNb;

    if ( id > sesdf->data.concavePatchNb )
        return;

    const float4 * atoms              = sesdf->data.atoms;
    const float4 * probeNeighbors     = sesdf->data.concavePatchesNeighbors;
    const float    probeRadius        = sesdf->data.probeRadius;
    const uint32_t maxProbeNeighborNb = sesdf->data.maxConcaveNeighbors;

    const float3   position        = make_float3( sesdf->data.concavePatchesPosition[ id ] );
    const int4     atomIds         = sesdf->data.concavePatchesId[ id ];
    const uint32_t neighborNb      = atomIds.w;
    const uint32_t startNeighborId = id * maxProbeNeighborNb;

    float3       ro = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );

    // Improving Numerical Precision in Intersection Programs, Ingo Walds, Raytracing Gems 2
    const float distToCenter = length( ro - position );
    float       toAdd        = 0.f;
    if ( distToCenter > probeRadius )
        toAdd = ( distToCenter - probeRadius );
    ro += rd * toAdd;

    const float3 oc = ro - position;
    const float  b  = dot( oc, rd );
    const float  c  = dot( oc, oc ) - probeRadius * probeRadius;
    const float  h  = b * b - c;

    if ( h >= 0.f )
    {
        const float4 atom1 = atoms[ atomIds.x ];
        const float3 p2    = make_float3( atom1 );
        const float4 atom2 = atoms[ atomIds.y ];
        const float3 p3    = make_float3( atom2 );
        const float4 atom3 = atoms[ atomIds.z ];
        const float3 p4    = make_float3( atom3 );

        const float3 center      = ( p2 + p3 + p4 ) / 3.f;
        const float3 dirToCenter = normalize( center - position );

        const float3 p2p1 = p2 - position;
        const float3 p3p1 = p3 - position;
        const float3 p4p1 = p4 - position;

        float3 n1 = normalize( cross( p2p1, p3p1 ) );
        float3 n2 = normalize( cross( p2p1, p4p1 ) );
        float3 n3 = normalize( cross( p3p1, p4p1 ) );

        n1 *= rvtx::cuda::sign( dot( dirToCenter, n1 ) );
        n2 *= rvtx::cuda::sign( dot( dirToCenter, n2 ) );
        n3 *= rvtx::cuda::sign( dot( dirToCenter, n3 ) );

        const float d1 = dot( position, n1 );
        const float d2 = dot( position, n2 );
        const float d3 = dot( position, n3 );

        const float sqrtH = sqrtf( h );

        const float tMin = optixGetRayTmin();
        const float tMax = optixGetRayTmax();

        float3      hit;
        float       t   = -1.f;
        const float tS1 = -b + sqrtH;
        if ( tS1 + toAdd > tMin && tS1 + toAdd < tMax
             && isInConcavePatch( hit = ro + tS1 * rd,
                                  startNeighborId,
                                  neighborNb,
                                  n1,
                                  d1,
                                  n2,
                                  d2,
                                  n3,
                                  d3,
                                  probeRadius,
                                  probeNeighbors ) )
        {
            t = tS1;
        }

        const float tS2 = -b - sqrtH;
        float3      temp;
        if ( tS2 + toAdd > tMin && tS2 + toAdd < tMax && ( t == -1.f || tS2 <= t )
             && isInConcavePatch( temp = ro + tS2 * rd,
                                  startNeighborId,
                                  neighborNb,
                                  n1,
                                  d1,
                                  n2,
                                  d2,
                                  n3,
                                  d3,
                                  probeRadius,
                                  probeNeighbors ) )
        {
            hit = temp;
            t   = tS2;
        }

        if ( t != -1.f )
        {
            t += toAdd;
            if ( t >= tMin && t <= tMax )
            {
                const float3 normal = ( position - hit ) / probeRadius;
                optixReportIntersection( t, 0, float3_as_ints( normal ), 0.f );
            }
        }
    }
}

extern "C" __global__ void __closesthit__concave()
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

    float3 position   = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    hitInfo->position = position;
    hitInfo->hit      = true;

    const auto *   hitGroup  = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   materials = hitGroup->materials;
    const auto *   sesdf     = reinterpret_cast<const rvtx::optix::SesdfHitGroupData *>( hitGroup->userPtr );
    const uint32_t id        = optixGetPrimitiveIndex() - sesdf->data.convexPatchNb;
    const int4     atomIds   = sesdf->data.concavePatchesId[ id ];

    const float3 p0 = make_float3( sesdf->data.concavePatchesPosition[ id ] );

    const float4 * atoms = sesdf->data.atoms;
    const float4   a1    = atoms[ atomIds.x ];
    const float4   a2    = atoms[ atomIds.y ];
    const float4   a3    = atoms[ atomIds.z ];
    const float3   p1    = make_float3( a1 );
    const float3   p2    = make_float3( a2 );
    const float3   p3    = make_float3( a3 );

    // Reference:
    // https://github.com/UniStuttgart-VISUS/megamol/blob/master/plugins/protein_cuda/shaders/protein_cuda/molecule_cb/mcbc_sphericaltriangle.frag.glsl#L124
    const float3 v1 = p1 - p0;
    const float3 v2 = p2 - p0;
    const float3 v3 = p3 - p0;

    const float3 u = v1 - v2;
    const float3 v = v3 - v2;
    // base point and direction of ray from the origin to the intersection point
    const float3 w    = make_float3( 0.f ) - v2;
    const float3 dRay = normalize( optixTransformPointFromWorldToObjectSpace( position ) - p0 );

    // cross products for computing the determinant
    const float3 wXu = cross( w, u );
    const float3 dXv = cross( dRay, v );
    // compute interse determinant
    float invdet = 1.0 / dot( dXv, u );
    // compute weights
    float beta   = dot( dXv, w ) * invdet;
    float gamma  = dot( wXu, dRay ) * invdet;
    float alpha2 = 1.0 - ( beta + gamma );

    if ( alpha2 > beta && alpha2 > gamma )
        hitInfo->material = materials[ atomIds.y ];
    else if ( beta > alpha2 && beta > gamma )
        hitInfo->material = materials[ atomIds.x ];
    else
        hitInfo->material = materials[ atomIds.z ];

    hitInfo->material.baseColor = materials[ atomIds.y ].baseColor * alpha2 + materials[ atomIds.x ].baseColor * beta
                                  + materials[ atomIds.z ].baseColor * gamma;
}

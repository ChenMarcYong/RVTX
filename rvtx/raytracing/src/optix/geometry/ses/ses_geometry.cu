#include <bcs/cuda/circles.cuh>
#include <device_launch_parameters.h>

#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/geometry/ses/ses_geometry.cuh"
#include "rvtx/optix/material/material.cuh"

namespace rvtx::optix
{
    __host__ void fillPatches( bcs::sesdf::SesdfData    data,
                               OptixAabb *              result,
                               uint32_t *               sbtIndex,
                               CircleBoundingGeometry * circlesBb,
                               CircleBoundingGeometry * segmentsBb )
    {
        std::size_t offset = 0;

        if ( data.convexPatchNb > 0 )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( data.convexPatchNb, 256 );
            fillConvexPatches<<<numBlocks, numThreads>>>( data, result + offset, sbtIndex + offset );
            offset += data.convexPatchNb;
        }

        if ( data.concavePatchNb )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( data.concavePatchNb, 256 );
            fillConcavePatches<<<numBlocks, numThreads>>>( data, result + offset, sbtIndex + offset );
            offset += data.concavePatchNb;
        }

        if ( data.circlePatchNb > 0 )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( data.circlePatchNb, 256 );
            fillCirclePatches<<<numBlocks, numThreads>>>( data, result + offset, sbtIndex + offset, circlesBb );
            offset += data.circlePatchNb;
        }

        if ( data.segmentPatchNb )
        {
            auto [ numBlocks, numThreads ] = cuda::KernelConfig::From( data.segmentPatchNb, 256 );
            fillSegmentPatches<<<numBlocks, numThreads>>>( data, result + offset, sbtIndex + offset, segmentsBb );
            offset += data.segmentPatchNb;
        }
    }

    inline __host__ __device__ float3 orthogonalVector( const float3 & normal )
    {
        float3 ref = { 1.f, 0.f, 0.f };
        if ( fabsf( dot( normal, ref ) ) > 1.f - 1e-4f )
            ref = make_float3( 0.f, 1.f, 0.f );
        return normalize( cross( normal, ref ) );
    }

    inline __device__ void getArcBoundingBox( const float3 v1,
                                              const float3 v2,
                                              const float3 n,
                                              float        maxAngle,
                                              float3 &     sMin,
                                              float3 &     sMax )
    {
        // Based on: https://stackoverflow.com/a/2618772
        const float3 right = normalize( cross( v1, n ) );
        sMin               = fminf( v1, v2 );
        sMax               = fmaxf( v1, v2 );

        float  angle = atan2f( right.x, v1.x );
        float3 v     = normalize( cosf( angle ) * v1 + sinf( angle ) * right );
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMax = fmaxf( sMax, v );

        v = -v;
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMin = fminf( sMin, v );

        angle = atan2f( right.y, v1.y );
        v     = normalize( cosf( angle ) * v1 + sinf( angle ) * right );
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMax = fmaxf( sMax, v );

        v = -v;
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMin = fminf( sMin, v );

        angle = atan2f( right.z, v1.z );
        v     = normalize( cosf( angle ) * v1 + sinf( angle ) * right );
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMax = fmaxf( sMax, v );

        v = -v;
        if ( bcs::angleBetweenEdges( v1, v, n ) < maxAngle )
            sMin = fminf( sMin, v );
    }

    __global__ void fillConvexPatches( bcs::sesdf::SesdfData data, OptixAabb * result, uint32_t * sbtIndex )
    {
        const uint32_t convexPatchId = blockIdx.x * blockDim.x + threadIdx.x;
        if ( convexPatchId >= data.convexPatchNb )
            return;

        sbtIndex[ convexPatchId ] = SesdfHitGroupData::SbtIndexConvex;

        uint2 ids = data.convexPatches[ convexPatchId ];

        const float4 atom    = data.atoms[ convexPatchId ];
        const float3 atomPos = make_float3( atom );

        float3 currentMin = atomPos - atom.w;
        float3 currentMax = atomPos + atom.w;

        for ( uint32_t i = ids.x; i < ids.y; i++ )
        {
            const float4 sector = data.sectors[ i ];
            const float3 sPos   = atomPos + make_float3( sector );
            currentMin          = fminf( currentMin, sPos );
            currentMax          = fmaxf( currentMax, sPos );
        }

        OptixAabb aabb;
        aabb.minX = currentMin.x;
        aabb.minY = currentMin.y;
        aabb.minZ = currentMin.z;
        aabb.maxX = currentMax.x;
        aabb.maxY = currentMax.y;
        aabb.maxZ = currentMax.z;

        result[ convexPatchId ] = aabb;
    }

    __global__ void fillConcavePatches( bcs::sesdf::SesdfData data, OptixAabb * result, uint32_t * sbtIndex )
    {
        const uint32_t intersectionId = blockIdx.x * blockDim.x + threadIdx.x;
        if ( intersectionId >= data.concavePatchNb )
            return;

        sbtIndex[ intersectionId ] = SesdfHitGroupData::SbtIndexConcave;

        const float4 intersection    = data.concavePatchesPosition[ intersectionId ];
        const int4   intersectionIds = data.concavePatchesId[ intersectionId ];

        const float4 atom1 = data.atoms[ intersectionIds.x ];
        const float4 atom2 = data.atoms[ intersectionIds.y ];
        const float4 atom3 = data.atoms[ intersectionIds.z ];

        const float3 p1 = make_float3( intersection );
        const float3 p2 = make_float3( atom1 );
        const float3 p3 = make_float3( atom2 );
        const float3 p4 = make_float3( atom3 );

        const float3 p2p1 = p2 - p1;
        const float3 p3p1 = p3 - p1;
        const float3 p4p1 = p4 - p1;

        const float3 triangleV1 = p2 - normalize( p2p1 ) * atom1.w;
        const float3 triangleV2 = p3 - normalize( p3p1 ) * atom2.w;
        const float3 triangleV3 = p4 - normalize( p4p1 ) * atom3.w;

        const float3 center       = ( p2 + p3 + p4 ) / 3.;
        const float3 dirToCenter  = normalize( center - p1 );
        const float3 deepestPoint = p1 + dirToCenter * data.probeRadius;

        const float3 vSphereCenter = ( deepestPoint + center ) * .5f;

        const float vSphereRadius
            = fmax( length( vSphereCenter - triangleV1 ),
                    fmax( length( vSphereCenter - triangleV2 ),
                          fmax( length( vSphereCenter - triangleV3 ), length( vSphereCenter - deepestPoint ) ) ) );

        const float3 currentMin = vSphereCenter - vSphereRadius;
        const float3 currentMax = vSphereCenter + vSphereRadius;

        OptixAabb aabb;
        aabb.minX = currentMin.x;
        aabb.minY = currentMin.y;
        aabb.minZ = currentMin.z;
        aabb.maxX = currentMax.x;
        aabb.maxY = currentMax.y;
        aabb.maxZ = currentMax.z;

        result[ intersectionId ] = aabb;
    }

    __global__ void fillCirclePatches( bcs::sesdf::SesdfData    data,
                                       OptixAabb *              result,
                                       uint32_t *               sbtIndex,
                                       CircleBoundingGeometry * circlesBb )
    {
        const uint32_t circlePatchId = blockIdx.x * blockDim.x + threadIdx.x;
        if ( circlePatchId >= data.circlePatchNb )
            return;

        sbtIndex[ circlePatchId ] = SesdfHitGroupData::SbtIndexCircle;

        const uint2  elementId = data.circlePatches[ circlePatchId ];
        const float4 atom1     = data.atoms[ elementId.x ];
        const float3 atom1Pos  = make_float3( atom1 );
        const float4 atom2     = data.atoms[ elementId.y ];
        const float3 atom2Pos  = make_float3( atom2 );

        const float3 circleCenter
            = bcs::getCircleCenter( atom1Pos, atom1.w + data.probeRadius, atom2Pos, atom2.w + data.probeRadius );

        const float3 circleToI             = atom1Pos - circleCenter;
        const float  squaredCircleDistance = dot( circleToI, circleToI );
        const float  ithExtendedRadius     = atom1.w + data.probeRadius;
        const float  radius                = sqrtf( ithExtendedRadius * ithExtendedRadius - squaredCircleDistance );

        const float3 iToJ   = atom2Pos - atom1Pos;
        const float3 normal = normalize( iToJ );
        const float3 p      = circleCenter + orthogonalVector( normal ) * radius;

        float3 x = normalize( p - atom1Pos ) * atom1.w;
        float3 c = ( length( p - atom1Pos ) / ( length( p - atom2Pos ) + length( p - atom1Pos ) ) )
                   * ( atom2Pos - atom1Pos );
        float d = length( x - c );
        c       = c + atom1Pos;

        const float4     sCircle     = bcs::getSmallCircle( atom1, normal, p );
        const float3     sCirclePos  = make_float3( sCircle );
        const float4     sCircle2    = bcs::getSmallCircle( atom2, normal, p );
        const float3     sCircle2Pos = make_float3( sCircle2 );
        constexpr float3 pMin = { -1.f, -1.f, 0.f }, pMax = { 1.f, 1.f, 0.f };
        const float      rad  = fmaxf( sCircle.w, sCircle2.w );
        float3           sMin = fminf( pMin * rad, pMin * fmaxf( 0.f, radius - data.probeRadius ) );
        float3           sMax = fmaxf( pMax * rad, pMax * fmaxf( 0.f, radius - data.probeRadius ) );
        const float2     dim  = fabs( make_float2( sMax - sMin ) * .5f );

        const float4 rotation = cuda::toLocalSpaceTransform( normal, { 0.f, 0.f, 1.f } );
        sMin                  = cuda::rotate( sMin, cuda::conjugate( rotation ) );
        sMax                  = cuda::rotate( sMax, cuda::conjugate( rotation ) );

        CircleBoundingGeometry circleBb;
        circleBb.rotation          = rotation;
        circleBb.bbDim             = make_float4( dim.x, dim.y, length( sCirclePos - sCircle2Pos ) * .5f, 0.f );
        circleBb.bbPos             = make_float4( ( sCirclePos + sCircle2Pos ) * .5f + ( sMax + sMin ) * .5f, 0.f );
        circleBb.vSphere           = make_float4( c, d );
        circlesBb[ circlePatchId ] = circleBb;

        const float3 currentMin = c - d;
        const float3 currentMax = c + d;
        OptixAabb    aabb;
        aabb.minX = currentMin.x;
        aabb.minY = currentMin.y;
        aabb.minZ = currentMin.z;
        aabb.maxX = currentMax.x;
        aabb.maxY = currentMax.y;
        aabb.maxZ = currentMax.z;

        result[ circlePatchId ] = aabb;
    }

    __global__ void fillSegmentPatches( bcs::sesdf::SesdfData    data,
                                        OptixAabb *              result,
                                        uint32_t *               sbtIndex,
                                        CircleBoundingGeometry * segmentsBb )
    {
        const uint32_t segmentId = blockIdx.x * blockDim.x + threadIdx.x;
        if ( segmentId >= data.segmentPatchNb )
            return;

        sbtIndex[ segmentId ] = SesdfHitGroupData::SbtIndexSegment;

        CircleBoundingGeometry segmentBb;
        const uint4            current = data.segmentPatches[ segmentId ];

        const uint32_t startAtomId = current.x;
        const uint32_t endAtomId   = current.y;

        const uint32_t startProbeId = current.z;
        const uint32_t endProbeId   = current.w;

        const float4 startAtom  = data.atoms[ startAtomId ];
        const float4 endAtom    = data.atoms[ endAtomId ];
        const float3 startProbe = make_float3( data.concavePatchesPosition[ startProbeId ] );
        const float3 endProbe   = make_float3( data.concavePatchesPosition[ endProbeId ] );

        const float3 startAtomPos = make_float3( startAtom );
        const float3 endAtomPos   = make_float3( endAtom );

        const float3 normal       = normalize( endAtomPos - startAtomPos );
        const float3 circleCenter = bcs::getCircleCenter(
            startAtomPos, startAtom.w + data.probeRadius, endAtomPos, endAtom.w + data.probeRadius );

        float3      v1           = ( startProbe - circleCenter );
        const float circleRadius = length( v1 );
        v1 /= circleRadius;
        const float3 v2 = ( endProbe - circleCenter ) / circleRadius;

        const float segmentAngle = bcs::angleBetweenEdges( v1, v2, normal );

        const float4 sCircle     = bcs::getSmallCircle( startAtom, normal, startProbe );
        const float3 sCirclePos  = make_float3( sCircle );
        const float4 sCircle2    = bcs::getSmallCircle( endAtom, normal, startProbe );
        const float3 sCircle2Pos = make_float3( sCircle2 );

        segmentBb.rotation = cuda::toLocalSpaceTransform( normal, make_float3( 0.f, 0.f, 1.f ) );
        float3 vMin, vMax;
        getArcBoundingBox( cuda::rotate( v1, segmentBb.rotation ),
                           cuda::rotate( v2, segmentBb.rotation ),
                           make_float3( 0.f, 0.f, 1.f ),
                           segmentAngle,
                           vMin,
                           vMax );

        float  rad  = fmax( sCircle.w, sCircle2.w );
        float3 sMin = fminf( vMin * rad, vMin * fmax( 0.f, circleRadius - data.probeRadius ) );
        float3 sMax = fmaxf( vMax * rad, vMax * fmax( 0.f, circleRadius - data.probeRadius ) );

        float2 dim      = fabs( make_float2( sMax - sMin ) * .5f );
        segmentBb.bbDim = make_float4( dim.x, dim.y, length( sCirclePos - sCircle2Pos ) * .5f, 1.f );

        float4 iRot = cuda::conjugate( segmentBb.rotation );
        sMin        = cuda::rotate( sMin, iRot );
        sMax        = cuda::rotate( sMax, iRot );

        segmentBb.bbPos = make_float4( ( sCirclePos + sCircle2Pos ) * .5f + ( sMax + sMin ) * .5f, segmentAngle );

        const float3 x = normalize( startProbe - startAtomPos ) * startAtom.w;
        float3       c = ( length( startProbe - startAtomPos )
                     / ( length( startProbe - endAtomPos ) + length( startProbe - startAtomPos ) ) )
                   * ( endAtomPos - startAtomPos );

        const float d           = length( x - c );
        c                       = c + startAtomPos;
        segmentBb.vSphere       = make_float4( c, d );
        segmentsBb[ segmentId ] = segmentBb;

        const float3 pMax       = sCircle2Pos + sMax;
        const float3 vSpherePos = ( pMax + ( sCirclePos + sMin ) ) * .5f;

        float        vSphereRadius = length( vSpherePos - pMax );
        const float3 currentMin    = vSpherePos - vSphereRadius;
        const float3 currentMax    = vSpherePos + vSphereRadius;

        OptixAabb aabb;
        aabb.minX = currentMin.x;
        aabb.minY = currentMin.y;
        aabb.minZ = currentMin.z;
        aabb.maxX = currentMax.x;
        aabb.maxY = currentMax.y;
        aabb.maxZ = currentMax.z;

        result[ segmentId ] = aabb;
    }
} // namespace rvtx::optix

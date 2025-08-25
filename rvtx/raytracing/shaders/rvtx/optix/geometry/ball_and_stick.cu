#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/optix/geometry/intersection.cuh>

#include "rvtx/optix/data.cuh"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_data.hpp"
#include "rvtx/optix/memory.cuh"

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

extern "C" __global__ void __intersection__sphere()
{
    const auto *   hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   bas      = reinterpret_cast<const rvtx::optix::BallAndStickHitGroupData *>( hitGroup->userPtr );
    const auto *   spheres  = bas->spheres;
    const uint32_t id       = optixGetPrimitiveIndex();

    const float4 sphere   = spheres[ id ];
    const float  radius   = sphere.w;
    const float3 position = make_float3( sphere.x, sphere.y, sphere.z );

    float3       ro = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );

    // Improving Numerical Precision in Intersection Programs, Ingo Walds, Raytracing Gems 2
    const float distToCenter = length( position - ro );
    float       toAdd        = 0.f;
    if ( distToCenter > radius )
        toAdd = distToCenter - radius;
    ro += rd * toAdd;

    const float2 intersections = rvtx::optix::iSphere( ro, rd, sphere );
    if ( intersections.y == -1.f )
        return;

    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();
    float t = intersections.x < intersections.y && intersections.x + toAdd >= tMin ? intersections.x : intersections.y;
    const float3 hit    = ro + rd * t;
    const float3 normal = normalize( hit - position );

    t += toAdd;
    if ( tMin < t && t < tMax )
        optixReportIntersection( t, 0, float3_as_ints( normal ), __float_as_int( radius ) );
}

extern "C" __global__ void __intersection__bond()
{
    const auto *   hitGroup = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   bas      = reinterpret_cast<const rvtx::optix::BallAndStickHitGroupData *>( hitGroup->userPtr );
    const uint32_t id       = optixGetPrimitiveIndex() - bas->sphereNb;

    const uint2  ids       = bas->bonds[ id ];
    const float4 startAtom = bas->spheres[ ids.x ];
    const float3 start     = make_float3( startAtom );
    const float4 endAtom   = bas->spheres[ ids.y ];
    const float3 end       = make_float3( endAtom );
    const float  radius    = bas->bondRadius;

    float3       ro = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 rd = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );

    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    // Reference: https://www.shadertoy.com/view/MtcXRf
    // The MIT License
    // Copyright © 2016 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    // documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    // the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    // to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
    // copyright notice and this permission notice shall be included in all copies or substantial portions of the
    // Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    // LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
    // SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
    // OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    // DEALINGS IN THE SOFTWARE. center the cylinder, normalize axis
    float3 cc = 0.5 * ( start + end );
    float  ch = length( end - start );
    float3 ca = ( end - start ) / ch;
    ch *= 0.5;

    float3 oc = ro - cc;

    float card = dot( ca, rd );
    float caoc = dot( ca, oc );

    float a = 1.0 - card * card;
    float b = dot( oc, rd ) - caoc * card;
    float c = dot( oc, oc ) - caoc * caoc - radius * radius;
    float h = b * b - a * c;
    if ( h < 0.0 )
        return;

    h        = ::sqrtf( h );
    float t1 = ( -b - h ) / a;
    float t2 = ( -b + h ) / a; // exit point

    // // caps
    // float sy = vtx::sign( y );
    // float tp = ( sy * ch - caoc ) / card;
    // if ( fabs( b + a * tp ) < h )
    // {
    //     const float3 normal = ca * sy;
    //     return make_float4( tp, normal.x, normal.y, normal.z );
    // }

    // body
    const float t = t1 < t2 && t1 >= tMin ? t1 : t2;
    float       y = caoc + t * card;

    if ( t > tMin && t < tMax && fabs( y ) < ch )
    {
        const float3 hit = ro + rd * t;
        if ( length( hit - start ) - startAtom.w < 0.f || length( hit - end ) - endAtom.w < 0.f )
            return;

        const float3 normal = normalize( oc + t * rd - ca * y );
        if ( tMin < t && t < tMax )
            optixReportIntersection( t, 0, float3_as_ints( normal ), __float_as_int( radius ) );
    }
}

extern "C" __global__ void __closesthit__sphere()
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

    const uint32_t id        = optixGetPrimitiveIndex();
    const auto *   hitGroup  = reinterpret_cast<rvtx::optix::GeometryHitGroup *>( optixGetSbtDataPointer() );
    const auto *   materials = hitGroup->materials;
    hitInfo->material        = materials[ id ];
}

extern "C" __global__ void __closesthit__bond()
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
    const auto * bas       = reinterpret_cast<const rvtx::optix::BallAndStickHitGroupData *>( hitGroup->userPtr );
    const auto * materials = hitGroup->materials;

    const uint32_t id    = optixGetPrimitiveIndex() - bas->sphereNb;
    const uint2    ids   = bas->bonds[ id ];
    const float3   start = make_float3( bas->spheres[ ids.x ] );
    const float3   end   = make_float3( bas->spheres[ ids.y ] );

    const float d1 = length( start - position );
    const float d2 = length( end - position );
    if ( d1 < d2 )
    {
        hitInfo->material = materials[ ids.x ];
    }
    else
    {
        hitInfo->material = materials[ ids.y ];
    }
}

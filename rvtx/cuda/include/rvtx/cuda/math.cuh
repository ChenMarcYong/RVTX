#ifndef RVTX_CUDA_MATH_CUH
#define RVTX_CUDA_MATH_CUH

#include <cstdint>

#include "helper_math.h"

namespace rvtx::cuda
{
    inline __host__ __device__ float sign( float f )
    {
        const float val = f > 0.f;
        return val - static_cast<float>( f < 0.f );
    }

    inline __host__ __device__ float length2( const float2 & v ) { return dot( v, v ); }
    inline __host__ __device__ float length2( const float3 & v ) { return dot( v, v ); }
    inline __host__ __device__ float length2( const float4 & v ) { return dot( v, v ); }

    inline __host__ __device__ float3 sqrtf( float3 a )
    {
        return make_float3( ::sqrtf( a.x ), ::sqrtf( a.y ), ::sqrtf( a.z ) );
    }

    inline __host__ __device__ float3 expf( const float3 & v )
    {
        return make_float3( ::expf( v.x ), ::expf( v.y ), ::expf( v.z ) );
    }

    inline __host__ __device__ float3 powf( const float3 & v, const float3 & e )
    {
        return make_float3( ::powf( v.x, e.x ), ::powf( v.y, e.y ), ::powf( v.z, e.z ) );
    }

    inline __host__ __device__ float determinant( const float3 a, const float3 b, const float3 c )
    {
        return dot( a, cross( b, c ) );
    }

    // Both n and ref must be normalized
    inline __device__ float4 toLocalSpaceTransform( const float3 n, const float3 ref )
    {
        //( dot( n, up ) < -1.f + 1e-4f ) // check if n is parallel to up
        if ( dot( n, ref ) < -1.f + 1e-4f )
            return make_float4( 1.f, 0.f, 0.f, 0.f );

        const float angle = 1.f + dot( n, ref ); // sqrt(length2(n) * length2(ref)) + dot( input, up );

        const float3 axis = cross( n, ref );
        return normalize( make_float4( axis, angle ) );
    }
    inline __device__ float4 toLocalZ( const float3 n ) { return toLocalSpaceTransform( n, { 0.f, 0.f, 1.f } ); }

    inline __device__ float4 conjugate( const float4 quat ) { return { -quat.x, -quat.y, -quat.z, quat.w }; }

    // Using GLM's way to multiply a vector with a quaternion
    inline __device__ float3 rotate( const float3 v, const float4 q )
    {
        const float3 qAxis = make_float3( q.x, q.y, q.z );
        return 2.0f * dot( qAxis, v ) * qAxis + ( q.w * q.w - dot( qAxis, qAxis ) ) * v
               + 2.0f * q.w * cross( qAxis, v );
    }

    inline __device__ __host__ uint8_t quantizeUnsigned8Bits( float x )
    {
        x = clamp( x, 0.0f, 1.0f );
        enum
        {
            N   = ( 1 << 8 ) - 1,
            Np1 = ( 1 << 8 )
        };
        return static_cast<uint8_t>( min( static_cast<int>( x * static_cast<float>( Np1 ) ), N ) );
    }

    // CUDA implementation of https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/refract.xhtml
    inline __host__ __device__ float3 refract( float3 i, float3 n, float eta )
    {
        const float k = 1.f - eta * eta * ( 1.f - dot( n, i ) * dot( n, i ) );
        if ( k < 0.f )
            return make_float3( 0.f );

        return eta * i - ( eta * dot( n, i ) + ::sqrtf( k ) ) * n;
    }
} // namespace rvtx::cuda

#endif // RVTX_CUDA_MATH_CUH
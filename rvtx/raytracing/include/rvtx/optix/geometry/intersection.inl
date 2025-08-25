#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/geometry/intersection.cuh"

namespace rvtx::optix
{
    // Inigo Quilez
    // ray-box intersection
    inline __device__ float2 iBox( const float3 ro, const float3 rd, const float3 pMin, const float3 pMax )
    {
        const float3 cen = .5f * ( pMin + pMax );
        const float3 rad = .5f * ( pMax - pMin );

        const float3 m = 1.f / rd;
        float3       n = m * ( ro - cen );
        float3       k = fabs( m ) * rad;

        const float3 t1 = -n - k;
        const float3 t2 = -n + k;

        float tN = fmaxf( fmaxf( t1.x, t1.y ), t1.z );
        float tF = fminf( fminf( t2.x, t2.y ), t2.z );

        if ( tN > tF || tF < 1e-4f )
            return make_float2( -1.f, -1.f );

        return make_float2( tN, tF );
    }

    inline __device__ float2 iOOBB( float3 ro, float3 rd, const float3 p, float3 dim, float4 rot )
    {
        ro -= p;
        ro = cuda::rotate( ro, rot ), rd = cuda::rotate( rd, rot );
        return iBox( ro, rd, -dim, dim );
    }

    inline __device__ float2 iSphere( const float3 ro, const float3 rd, const float4 sph )
    {
        const float3 oc = ro - make_float3( sph );
        const float  b  = dot( oc, rd );
        const float  c  = dot( oc, oc ) - sph.w * sph.w;
        const float  h  = b * b - c;
        if ( h < 0.f )
            return make_float2( -1.f, -1.f );
        return make_float2( -b - ::sqrtf( h ), -b + ::sqrtf( h ) );
    }
} // namespace rvtx::optix

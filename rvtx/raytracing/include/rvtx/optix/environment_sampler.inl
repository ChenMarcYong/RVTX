#include <helper_math.h>

#include "rvtx/core/math.hpp"
#include "rvtx/optix/environment_sampler.cuh"

namespace rvtx::optix
{
    __device__ inline float4 getEnvironmentColor( const Texture::View & environment, const float3 v, const float lod )
    {
        const float theta = ::acosf( ::clamp( v.y, -1.f, 1.f ) );

        float phi = copysignf( 1.f, v.z ) * ::acosf( v.x / length( make_float2( v.x, v.z ) ) );
        phi       = phi < 0.f ? phi + 2.f * rvtx::Pi : phi;

        const float2 uv = make_float2( phi / ( 2.f * Pi ), theta / Pi );
        return environment.get<float4>( uv.x, uv.y, lod );
    }

    __device__ inline float2 sample2D( const Texture::View & environment, float2 u, float & pdf )
    {
        const int   maxMipMap = static_cast<float>( environment.lodLevels );
        const float fwidth    = static_cast<float>( environment.width );
        const float fheight   = static_cast<float>( environment.height );

        int x = 0, y = 0;
        for ( int mip = maxMipMap; mip >= 0; --mip )
        {
            x <<= 1;
            y <<= 1;

            const float a = environment.get<float4>( float( x ) / fwidth, float( y ) / fheight, float( mip ) ).x;
            const float b = environment.get<float4>( float( x ) / fwidth, float( y + 1 ) / fheight, float( mip ) ).x;
            const float c = environment.get<float4>( float( x + 1 ) / fwidth, float( y ) / fheight, float( mip ) ).x;
            const float d
                = environment.get<float4>( float( x + 1 ) / fwidth, float( y + 1 ) / fheight, float( mip ) ).x;

            const float left     = a + b;
            const float right    = c + d;
            const float probLeft = left / ( left + right );
            if ( u.x < probLeft )
            {
                u.x /= probLeft;
                float probLower = a / left;
                if ( u.y < probLower )
                {
                    u.y /= probLower;
                }
                else
                {
                    y++;
                    u.y = ( u.y - probLower ) / ( 1.f - probLower );
                }
            }
            else
            {
                x++;
                u.x             = ( u.x - probLeft ) / ( 1.f - probLeft );
                float probLower = a / right;
                if ( u.y < probLower )
                {
                    u.y /= probLower;
                }
                else
                {
                    y++;
                    u.y = ( u.y - probLower ) / ( 1.f - probLower );
                }
            }
        }

        pdf = environment.get<float4>( float( x ) / fwidth, float( y ) / fheight ).x
              / environment.get<float4>( 0.f, 0.f, static_cast<float>( maxMipMap ) ).x;

        const float size = ::fmaxf( static_cast<float>( environment.width ), static_cast<float>( environment.height ) );
        return make_float2( float( x ), float( y ) ) / size;
    }

    __device__ inline float3 sampleEnvironment( const Texture::View & environment, float2 u, float & pdf )
    {
        float2 uv = sample2D( environment, u, pdf );

        // We want X to be mapped from 0 to 2Pi since that's where the image is the largest
        const float theta = uv.y * Pi;
        const float phi   = uv.x * 2.f * Pi;

        const float cosTheta = cos( theta );
        const float sinTheta = sin( theta );
        const float cosPhi   = cos( phi );
        const float sinPhi   = sin( phi );

        pdf /= fmaxf( 1e-4f,
                      2.f * rvtx::Pi * rvtx::Pi // Density in terms of spherical coordinates
                          * sinTheta            // Mapping jacobian
        );

        return normalize( make_float3( sinTheta * cosPhi, cosTheta, sinTheta * sinPhi ) );
    }

    __device__ inline float getPdfEnvironment( const Texture::View & environment, float3 v )
    {
        const float theta    = acosf( clamp( v.y, -1.f, 1.f ) );
        const float sinTheta = sinf( theta );

        float phi = copysignf( 1.f, v.z ) * acos( v.x / length( make_float2( v.x, v.z ) ) );
        phi       = phi < 0.f ? phi + 2.f * rvtx::Pi : phi;

        const float2 uv  = make_float2( phi / ( 2.f * rvtx::Pi ), theta / rvtx::Pi );
        float        pdf = environment.get<float4>( uv.x, uv.y, 0.f ).x
                    / environment.get<float4>( uv.x, uv.y, environment.lodLevels ).x;
        pdf /= fmaxf( 1e-4f,
                      2. * Pi * Pi   // Density in terms of spherical coordinates
                          * sinTheta // Mapping jacobian
        );

        return pdf;
    }
} // namespace rvtx::optix
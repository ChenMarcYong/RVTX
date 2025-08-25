#include <helper_math.h>

#include "rvtx/core/math.hpp"
#include "rvtx/cuda/math.cuh"
#include "rvtx/optix/material/color.cuh"

namespace rvtx::optix
{
    inline __device__ __host__ float3 linearToSRGB( const float3 color, const float gamma )
    {
        return { pow( color.x, gamma ), pow( color.y, gamma ), pow( color.z, gamma ) };
    }
    inline __device__ __host__ float4 linearToSRGB( const float4 color, const float gamma )
    {
        return { pow( color.x, gamma ), pow( color.y, gamma ), pow( color.z, gamma ), color.w };
    }

    inline __device__ __host__ uchar4 float3ToColor( float3 color )
    {
        color = clamp( color, 0.f, 1.f );
        return make_uchar4( cuda::quantizeUnsigned8Bits( color.x ),
                            cuda::quantizeUnsigned8Bits( color.y ),
                            cuda::quantizeUnsigned8Bits( color.z ),
                            255u );
    }

    inline __device__ __host__ uchar4 float4ToColor( float4 color )
    {
        color = clamp( color, 0.f, 1.f );
        return make_uchar4( cuda::quantizeUnsigned8Bits( color.x ),
                            cuda::quantizeUnsigned8Bits( color.y ),
                            cuda::quantizeUnsigned8Bits( color.z ),
                            color.w * 255u );
    }

    // Ref: https://en.wikipedia.org/wiki/Relative_luminance
    inline __host__ __device__ float getLuminance( const float3 rgb )
    {
        return rgb.x * 0.2126f + rgb.y * 0.7152f + rgb.z * 0.0722f;
    }

    // ACES Filmic Tone Mapping Curve, Krzysztof Narkowicz, 2016
    // Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    inline __device__ float3 ACESFilm( const float3 x )
    {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        return clamp( ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e ), make_float3( 0.f ), make_float3( 1.f ) );
    }

    inline __device__ float3 fadedBackground( const float3 rd )
    {
        constexpr float3 palette[ 3 ] = {
            float3 { 0.445f, 0.625f, 0.892f },
            float3 { 0.573f, 0.467f, 0.549f },
            float3 { 1.f, 0.478f, 0.141f },
        };

        float angle = acosf( rd.y );

        float3 color = lerp( palette[ 0 ], palette[ 2 ], fabs( angle ) / Pi );
        color        = cuda::powf( color, make_float3( 1.5f ) );

        return color;
    }
} // namespace rvtx::optix

#include "rvtx/optix/environment.cuh"
#include "rvtx/optix/material/color.cuh"

namespace rvtx::optix
{
    std::vector<float> getSamplingImage( uint32_t               width,
                                         uint32_t               height,
                                         uint32_t               channels,
                                         rvtx::ConstSpan<float> data )
    {
        assert( channels >= 3 && "This function is intended to be used for 3 or 4 channels images." );
        std::vector<float> result {};

        const uint32_t samplingSize = std::min( width, height );
        result.resize( samplingSize * samplingSize * 4 );

        const uint32_t xStepSize = width / samplingSize;
        const uint32_t yStepSize = height / samplingSize;
        for ( uint32_t y = 0; y < height; y += yStepSize )
        {
            // Weighting term to avoid too much sampling on the pole
            const float sinTheta
                = std::sin( rvtx::Pi * ( static_cast<float>( y ) + .5f ) / static_cast<float>( height ) );
            for ( uint32_t x = 0; x < width; x += xStepSize )
            {
                const uint32_t pixel = ( y * width + x ) * 4u;

                const float r = data[ pixel + 0u ];
                const float g = data[ pixel + 1u ];
                const float b = data[ pixel + 2u ];

                const uint32_t xx = x / xStepSize;
                const uint32_t yy = y / yStepSize;

                const float weightedLuminance = rvtx::optix::getLuminance( make_float3( r, g, b ) ) * sinTheta;

                const uint32_t ppixel = ( yy * samplingSize + xx ) * channels;
                result[ ppixel + 0u ] = weightedLuminance;
                result[ ppixel + 1u ] = weightedLuminance;
                result[ ppixel + 2u ] = weightedLuminance;
                result[ ppixel + 3u ] = weightedLuminance;
            }
        }

        return result;
    }
} // namespace rvtx::optix
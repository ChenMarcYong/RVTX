#include "rvtx/optix/environment.hpp"

#include <rvtx/core/image.hpp>
#include <rvtx/optix/environment_sampler.cuh>

namespace rvtx::optix
{
    Environment::Environment( std::filesystem::path path, float weight )
    {
        uint32_t           width, height, channels;
        std::vector<float> pixels = rvtx::loadExr( path, width, height, channels );

        for ( uint32_t pixel = 0; pixel < height * width; pixel++ )
        {
            const float r = pixels[ pixel * channels + 0u ];
            const float g = pixels[ pixel * channels + 1u ];
            const float b = pixels[ pixel * channels + 2u ];

            const glm::vec3 color = glm::pow( { r, g, b }, glm::vec3 { 1.f / 2.2f } );

            pixels[ pixel * channels + 0u ] = color.r * weight;
            pixels[ pixel * channels + 1u ] = color.g * weight;
            pixels[ pixel * channels + 2u ] = color.b * weight;
        }

        const uint32_t           samplingSize = std::min( width, height );
        const std::vector<float> samplingImg  = rvtx::optix::getSamplingImage( width, height, channels, pixels );

        m_environment = rvtx::optix::Texture::From( width, height, pixels );
        m_sampling    = rvtx::optix::Texture::From( samplingSize, samplingSize, samplingImg );
    }
} // namespace rvtx::optix
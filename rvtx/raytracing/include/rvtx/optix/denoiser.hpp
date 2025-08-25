#ifndef RVTX_OPTIX_DENOISER_HPP
#define RVTX_OPTIX_DENOISER_HPP

#include <cstdint>

#include <optix_types.h>

#include "rvtx/optix/context.cuh"

namespace rvtx::optix
{
    // Based on:https://github.com/nvpro-samples/vk_denoise/blob/master/optix_denoiser/src/denoiser.hpp
    class Denoiser
    {
      public:
        Denoiser( const Context &  context,
                  uint32_t         width,
                  uint32_t         height,
                  OptixPixelFormat pixelFormat,
                  bool             withAlbedo = false,
                  bool             withNormal = false );

        Denoiser( const Denoiser & )             = delete;
        Denoiser & operator=( const Denoiser & ) = delete;

        Denoiser( Denoiser && ) noexcept;
        Denoiser & operator=( Denoiser && ) noexcept;

        ~Denoiser();

        template<class PixelType>
        void denoise( PixelType * input,
                      PixelType * output,
                      PixelType * albedo = nullptr,
                      PixelType * normal = nullptr );

      private:
        const Context *      m_context = nullptr;
        OptixDenoiser        m_denoiser { nullptr };
        OptixDenoiserOptions m_dOptions {};
        uint32_t             m_width;
        uint32_t             m_height;

        OptixDenoiserSizes m_dSizes {};
        CUdeviceptr        m_dStateBuffer { 0 };
        CUdeviceptr        m_dScratchBuffer { 0 };
        CUdeviceptr        m_dIntensity { 0 };
        CUdeviceptr        m_dMinRGB { 0 };

        OptixPixelFormat m_pixelFormat;
        uint32_t         m_sizeofPixel {};
    };
} // namespace rvtx::optix

#include "rvtx/optix/denoiser.inl"

#endif // RVTX_OPTIX_DENOISER_HPP

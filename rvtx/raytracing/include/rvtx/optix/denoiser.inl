#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/setup.cuh"
#include "rvtx/optix/denoiser.hpp"

namespace rvtx::optix
{
    template<class PixelType>
    void Denoiser::denoise( PixelType * input, PixelType * output, PixelType * albedo, PixelType * normal )
    {
        OptixPixelFormat pixelFormat      = m_pixelFormat;
        auto             sizeofPixel      = m_sizeofPixel;
        uint32_t         rowStrideInBytes = sizeofPixel * m_width;

        // Create and set our OptiX layers
        OptixDenoiserLayer layer = {};
        // Input
        layer.input.data               = reinterpret_cast<CUdeviceptr>( input );
        layer.input.width              = m_width;
        layer.input.height             = m_height;
        layer.input.rowStrideInBytes   = rowStrideInBytes;
        layer.input.pixelStrideInBytes = m_sizeofPixel;
        layer.input.format             = pixelFormat;

        // Output
        layer.output.data               = reinterpret_cast<CUdeviceptr>( output );
        layer.output.width              = m_width;
        layer.output.height             = m_height;
        layer.output.rowStrideInBytes   = rowStrideInBytes;
        layer.output.pixelStrideInBytes = sizeof( PixelType );
        layer.output.format             = pixelFormat;

        OptixDenoiserGuideLayer guideLayer = {};

        if ( m_dOptions.guideAlbedo != 0u )
        {
            guideLayer.albedo.data               = reinterpret_cast<CUdeviceptr>( albedo );
            guideLayer.albedo.width              = m_width;
            guideLayer.albedo.height             = m_height;
            guideLayer.albedo.rowStrideInBytes   = rowStrideInBytes;
            guideLayer.albedo.pixelStrideInBytes = m_sizeofPixel;
            guideLayer.albedo.format             = pixelFormat;
        }

        // normal
        if ( m_dOptions.guideNormal != 0u )
        {
            guideLayer.normal.data               = reinterpret_cast<CUdeviceptr>( normal );
            guideLayer.normal.width              = m_width;
            guideLayer.normal.height             = m_height;
            guideLayer.normal.rowStrideInBytes   = rowStrideInBytes;
            guideLayer.normal.pixelStrideInBytes = m_sizeofPixel;
            guideLayer.normal.format             = pixelFormat;
        }

        if ( m_dIntensity != 0 )
        {
            optixCheck( optixDenoiserComputeIntensity( m_denoiser,
                                                       m_context->getStream(),
                                                       &layer.input,
                                                       m_dIntensity,
                                                       m_dScratchBuffer,
                                                       m_dSizes.withoutOverlapScratchSizeInBytes ) );
        }

        OptixDenoiserParams denoiserParams {};
        denoiserParams.hdrIntensity = m_dIntensity;
        denoiserParams.blendFactor  = 0.0F; // Fully denoised

        // Execute the denoiser
        optixCheck( optixDenoiserInvoke( m_denoiser,
                                         m_context->getStream(),
                                         &denoiserParams,
                                         m_dStateBuffer,
                                         m_dSizes.stateSizeInBytes,
                                         &guideLayer,
                                         &layer,
                                         1,
                                         0,
                                         0,
                                         m_dScratchBuffer,
                                         m_dSizes.withoutOverlapScratchSizeInBytes ) );

        cuda::cudaCheck( cudaStreamSynchronize( m_context->getStream() ) ); // Making sure the denoiser is done
    }
} // namespace rvtx::optix

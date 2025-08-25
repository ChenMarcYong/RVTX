#include "rvtx/optix/denoiser.hpp"

namespace rvtx::optix
{
    Denoiser::Denoiser( const Context &  context,
                        uint32_t         width,
                        uint32_t         height,
                        OptixPixelFormat pixelFormat,
                        bool             withAlbedo,
                        bool             withNormal ) :
        m_context( &context ),
        m_width( width ), m_height( height ), m_pixelFormat( pixelFormat )
    {
        switch ( pixelFormat )
        {
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            m_sizeofPixel  = static_cast<uint32_t>( 3 * sizeof( float ) );
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            m_sizeofPixel  = static_cast<uint32_t>( 4 * sizeof( float ) );
            break;
        case OPTIX_PIXEL_FORMAT_UCHAR3:
            m_sizeofPixel  = static_cast<uint32_t>( 3 * sizeof( uint8_t ) );
            break;
        case OPTIX_PIXEL_FORMAT_UCHAR4:
            m_sizeofPixel  = static_cast<uint32_t>( 4 * sizeof( uint8_t ) );
            break;
        case OPTIX_PIXEL_FORMAT_HALF3:
            m_sizeofPixel  = static_cast<uint32_t>( 3 * sizeof( uint16_t ) );
            break;
        case OPTIX_PIXEL_FORMAT_HALF4:
            m_sizeofPixel  = static_cast<uint32_t>( 4 * sizeof( uint16_t ) );
            break;
        default: logger::error( "Format not supported" ); break;
        }

        // This is to use RGB + Albedo + Normal
        m_dOptions.guideAlbedo            = withAlbedo;
        m_dOptions.guideNormal            = withNormal;
        OptixDenoiserModelKind model_kind = false ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
        model_kind                        = OPTIX_DENOISER_MODEL_KIND_AOV;
        optixCheck( optixDenoiserCreate( m_context->getOptiXContext(), model_kind, &m_dOptions, &m_denoiser ) );

        // Computing the amount of memory needed to do the denoiser
        optixCheck( optixDenoiserComputeMemoryResources( m_denoiser, m_width, m_height, &m_dSizes ) );

        cuda::cudaCheck( cudaMalloc( (void **)&m_dStateBuffer, m_dSizes.stateSizeInBytes ) );
        cuda::cudaCheck( cudaMalloc( (void **)&m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes ) );
        cuda::cudaCheck( cudaMalloc( (void **)&m_dMinRGB, 4 * sizeof( float ) ) );
        if ( m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT3 || m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT4 )
            cuda::cudaCheck( cudaMalloc( (void **)&m_dIntensity, sizeof( float ) ) );

        optixCheck( optixDenoiserSetup( m_denoiser,
                                        m_context->getStream(),
                                        m_width,
                                        m_height,
                                        m_dStateBuffer,
                                        m_dSizes.stateSizeInBytes,
                                        m_dScratchBuffer,
                                        m_dSizes.withoutOverlapScratchSizeInBytes ) );
    }

    Denoiser::Denoiser( Denoiser && other ) noexcept
    {
        std::swap( m_context, other.m_context );
        std::swap( m_denoiser, other.m_denoiser );
        std::swap( m_dOptions, other.m_dOptions );
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_dSizes, other.m_dSizes );
        std::swap( m_dStateBuffer, other.m_dStateBuffer );
        std::swap( m_dScratchBuffer, other.m_dScratchBuffer );
        std::swap( m_dIntensity, other.m_dIntensity );
        std::swap( m_dMinRGB, other.m_dMinRGB );
        std::swap( m_pixelFormat, other.m_pixelFormat );
        std::swap( m_sizeofPixel, other.m_sizeofPixel );
    }

    Denoiser & Denoiser::operator=( Denoiser && other ) noexcept
    {
        std::swap( m_context, other.m_context );
        std::swap( m_denoiser, other.m_denoiser );
        std::swap( m_dOptions, other.m_dOptions );
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_dSizes, other.m_dSizes );
        std::swap( m_dStateBuffer, other.m_dStateBuffer );
        std::swap( m_dScratchBuffer, other.m_dScratchBuffer );
        std::swap( m_dIntensity, other.m_dIntensity );
        std::swap( m_dMinRGB, other.m_dMinRGB );
        std::swap( m_pixelFormat, other.m_pixelFormat );
        std::swap( m_sizeofPixel, other.m_sizeofPixel );

        return *this;
    }

    Denoiser::~Denoiser()
    {
        if ( m_dStateBuffer != 0 )
        {
            cuda::cudaCheck( cudaFree( (void *)m_dStateBuffer ) );
            m_dStateBuffer = NULL;
        }
        if ( m_dScratchBuffer != 0 )
        {
            cuda::cudaCheck( cudaFree( (void *)m_dScratchBuffer ) );
            m_dScratchBuffer = NULL;
        }
        if ( m_dIntensity != 0 )
        {
            cuda::cudaCheck( cudaFree( (void *)m_dIntensity ) );
            m_dIntensity = NULL;
        }
        if ( m_dMinRGB != 0 )
        {
            cuda::cudaCheck( cudaFree( (void *)m_dMinRGB ) );
            m_dMinRGB = NULL;
        }
    }
} // namespace rvtx::optix

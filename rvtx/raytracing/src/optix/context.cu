#include "rvtx/core/logger.hpp"
#include "rvtx/cuda/setup.cuh"
#include "rvtx/optix/context.cuh"

namespace rvtx::optix
{
    static void contextLogCallback( unsigned int level, const char * tag, const char * message, void * )
    {
        logger::debug( "[OPTIX][{}][{}]: {}", level, tag, message );
    }

    Context::Context()
    {
        int numDevices;
        cudaGetDeviceCount( &numDevices );
        logger::debug( "Found {} CUDA devices", numDevices );

        optixCheck( optixInit() );
        // for this sample, do everything on one device
        const int deviceID = 0;
        cuda::cudaCheck( cudaSetDevice( deviceID ) );
        cuda::cudaCheck( cudaStreamCreate( &m_stream ) );

        cudaGetDeviceProperties( &m_deviceProps, deviceID );
        logger::debug( "Running on device: {}", m_deviceProps.name );

        CUcontext cuCtx = 0; // zero means take the current context
        optixCheck( optixInit() );
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &contextLogCallback;
        options.logCallbackLevel          = 4;
        options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        optixCheck( optixDeviceContextCreate( cuCtx, &options, &m_optiXContext ) );
    }

    Context::Context( Context && other ) noexcept
    {
        std::swap( m_stream, other.m_stream );
        std::swap( m_deviceProps, other.m_deviceProps );
        std::swap( m_optiXContext, other.m_optiXContext );
    }

    Context & Context::operator=( Context && other ) noexcept
    {
        std::swap( m_stream, other.m_stream );
        std::swap( m_deviceProps, other.m_deviceProps );
        std::swap( m_optiXContext, other.m_optiXContext );

        return *this;
    }

    Context::~Context() = default;
} // namespace rvtx::optix

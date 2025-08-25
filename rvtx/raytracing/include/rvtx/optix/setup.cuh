#ifndef RVTX_OPTIX_SETUP_CUH
#define RVTX_OPTIX_SETUP_CUH

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "rvtx/core/logger.hpp"

#define optixCheck( call )                                                                            \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if ( res != OPTIX_SUCCESS )                                                                   \
            rvtx::logger::debug( "[OPTIX]: {} failed with code {} (line {})", #call, res, __LINE__ ); \
    }

#define optixCheckLog( call )                                                                          \
    do                                                                                                 \
    {                                                                                                  \
        OptixResult  res                 = call;                                                       \
        const size_t sizeof_log_returned = sizeOfLog;                                                  \
        sizeOfLog                        = sizeof( log ); /* reset sizeof_log for future calls */      \
        if ( res != OPTIX_SUCCESS )                                                                    \
        {                                                                                              \
            rvtx::logger::debug( "[OPTIX]: {} failed with code {} (line {})", #call, res, __LINE__ );  \
            rvtx::logger::debug(                                                                       \
                "[OPTIX]: {} {}", log, ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) ); \
        }                                                                                              \
    } while ( 0 )

namespace rvtx::optix
{
    template<typename T>
    struct Record
    {
        alignas( OPTIX_SBT_RECORD_ALIGNMENT ) char header[ OPTIX_SBT_RECORD_HEADER_SIZE ];
        T data;
    };
} // namespace rvtx::optix

#endif // RVTX_OPTIX_SETUP_CUH

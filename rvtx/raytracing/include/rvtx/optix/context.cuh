#ifndef RVTX_OPTIX_CONTEXT_CUH
#define RVTX_OPTIX_CONTEXT_CUH

#include "rvtx/optix/setup.cuh"

namespace rvtx::optix
{
    class Context
    {
      public:
        Context();

        Context( const Context & )           = delete;
        Context operator=( const Context & ) = delete;

        Context( Context && other ) noexcept;
        Context & operator=( Context && other ) noexcept;

        ~Context();

        inline CUstream           getStream() const;
        inline OptixDeviceContext getOptiXContext() const;

      private:
        CUstream           m_stream = 0;
        cudaDeviceProp     m_deviceProps;
        OptixDeviceContext m_optiXContext;
    };
} // namespace rvtx::optix

#include "rvtx/optix/context.inl"

#endif // RVTX_OPTIX_CONTEXT_CUH
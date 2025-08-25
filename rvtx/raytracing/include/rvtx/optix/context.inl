#include "rvtx/optix/context.cuh"

namespace rvtx::optix
{
    inline CUstream           Context::getStream() const { return m_stream; }
    inline OptixDeviceContext Context::getOptiXContext() const { return m_optiXContext; }
} // namespace rvtx::optix
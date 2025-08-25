#include "rvtx/optix/geometry.cuh"

namespace rvtx::optix
{
    OptixTraversableHandle BaseGeometry::getGASHandle() const { return m_gasHandle; }
} // namespace rvtx::optix

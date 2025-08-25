#include "rvtx/optix/geometry.cuh"

namespace rvtx::optix
{
    BaseGeometry::BaseGeometry( const Context & context ) : m_context( &context ) {}
} // namespace rvtx::optix

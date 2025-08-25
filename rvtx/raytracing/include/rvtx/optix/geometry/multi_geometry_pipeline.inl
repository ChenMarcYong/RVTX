#include "rvtx/optix/geometry/multi_geometry_pipeline.hpp"

namespace rvtx::optix
{
    inline void MultiGeometryPipeline::setRayGen( Module & oModule, std::string name )
    {
        oModule.compile( *this );
        m_rayGenModule = std::make_pair( &oModule, name );
    }

    inline void MultiGeometryPipeline::setMiss( Module & oModule, std::string name )
    {
        oModule.compile( *this );
        m_missModule = std::make_pair( &oModule, name );
    }

    template<class DerivedGeometryHandler, class... Args>
    DerivedGeometryHandler & MultiGeometryPipeline::add( Args &&... args )
    {
        static_assert( std::is_base_of<GeometryHandler, DerivedGeometryHandler>::value,
                       "DerivedGeometryHandler must be based on GeometryHandler." );

        auto derivedHandler          = std::make_unique<DerivedGeometryHandler>( std::forward<Args>( args )... );
        DerivedGeometryHandler * ptr = derivedHandler.get();
        m_handles.emplace_back( std::move( derivedHandler ) );
        return *ptr;
    }

    inline OptixTraversableHandle MultiGeometryPipeline::getHandle() const { return m_handle; }
} // namespace rvtx::optix

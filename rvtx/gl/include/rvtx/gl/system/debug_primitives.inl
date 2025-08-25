#include "rvtx/gl/system/debug_primitives.hpp"
#include "rvtx/system/scene.hpp"

namespace rvtx::gl
{
    inline DebugPrimitives::DebugPrimitives( DebugPrimitivesHolder * holder ) : holder( holder ) {}

    inline DebugPrimitives::DebugPrimitives( Scene & scene, std::string entityName )
    {
        const auto entity = scene.createEntity( std::move( entityName ) );
        holder            = &entity.emplace<DebugPrimitivesHolder>();
    }

    /* --------------------------- ------- --------------------------- */
    /* --------------------------- SPHERES --------------------------- */

    inline void DebugPrimitives::addSphere( const Sphere & sphere ) { spheres.emplace_back( sphere ); }

    inline void DebugPrimitives::addSphere( const glm::vec3 & position,
                                            const glm::vec3 & color,
                                            const float       radius,
                                            const bool        visible )
    {
        addSphere( Sphere { position, color, radius, visible } );
    }

    /* --------------------------- ----- --------------------------- */
    /* --------------------------- LINES --------------------------- */

    inline void DebugPrimitives::addLine( const Line & line ) { lines.emplace_back( line ); }

    inline void DebugPrimitives::addLine( const glm::vec3 & start,
                                          const glm::vec3 & end,
                                          const glm::vec3 & color,
                                          const float       radius,
                                          const bool        capped )
    {
        addLine( Line { Sphere { start, color, radius }, Sphere { end, color, radius }, color, radius, capped } );
    }

    inline void DebugPrimitives::addLine( const Sphere &    start,
                                          const Sphere &    end,
                                          const float       radius,
                                          const glm::vec3 & color )
    {
        addLine( Line { start, end, color, radius } );
    }

    /* --------------------------- ----- --------------------------- */
    /* --------------------------- PATHS --------------------------- */

    inline void DebugPrimitives::addPath( const Path & path ) { paths.emplace_back( path ); }

    inline void DebugPrimitives::addPath( const rvtx::Path<glm::vec3> & path,
                                          const glm::vec3               pathColor,
                                          const float                   pathRadius,
                                          const uint32_t                segmentsCount )
    {
        addPath( Path { path, pathColor, pathRadius, segmentsCount, pathColor, pathRadius, true } );
    }

    inline void DebugPrimitives::addPath( const rvtx::Path<glm::vec3> & path,
                                          const glm::vec3               pathColor,
                                          const float                   pathRadius,
                                          const glm::vec3               segmentStepColor,
                                          const float                   segmentStepRadius,
                                          const uint32_t                segmentsCount,
                                          const bool                    segmentsVisible )
    {
        addPath(
            Path { path, pathColor, pathRadius, segmentsCount, segmentStepColor, segmentStepRadius, segmentsVisible } );
    }
} // namespace rvtx::gl

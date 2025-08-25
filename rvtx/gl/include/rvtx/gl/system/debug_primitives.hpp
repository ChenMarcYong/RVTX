#ifndef RVTX_GL_SYSTEM_DEBUG_PRIMITIVES_HPP
#define RVTX_GL_SYSTEM_DEBUG_PRIMITIVES_HPP

#include <vector>

#include <glm/vec3.hpp>

#include "rvtx/core/color.hpp"
#include "rvtx/core/path.hpp"
#include "rvtx/gl/geometry/debug_primitives_geometry.hpp"

namespace rvtx
{
    class Scene;
}

namespace rvtx::gl
{
    struct DebugPrimitives
    {
        struct Sphere
        {
            glm::vec3 position;
            glm::vec3 color { 1.f };
            float     radius { 1.f };
            bool      visible { true };
        };

        struct Line
        {
            Sphere    start;
            Sphere    end;
            glm::vec3 color { 1.f };
            float     radius { 0.75f };
            bool      capped { true };
        };

        struct Path
        {
            rvtx::Path<glm::vec3> path;
            glm::vec3             pathColor { 1.f };
            float                 pathRadius { 1.f };
            uint32_t              segmentsCount = 1000;
            glm::vec3             segmentStepColor { 1.f };
            float                 segmentStepRadius { 1.f };
            bool                  segmentsVisible { true };
        };

        DebugPrimitives() = default;
        DebugPrimitives( DebugPrimitivesHolder * holder );
        DebugPrimitives( Scene & scene, std::string entityName = "Unamed Debug Primitives" );
        ~DebugPrimitives() = default;

        void update();
        void clear( const bool updateHolder = false );

        DebugPrimitivesHolder * holder { nullptr };

        /* --------------------------- ------- --------------------------- */
        /* --------------------------- SPHERES --------------------------- */

        std::vector<Sphere> spheres;

        inline void addSphere( const Sphere & sphere );
        inline void addSphere( const glm::vec3 & position,
                               const glm::vec3 & color   = glm::vec3 { 1.f },
                               const float       radius  = 1.f,
                               const bool        visible = true );

        /* --------------------------- ----- --------------------------- */
        /* --------------------------- LINES --------------------------- */

        std::vector<Line> lines;
        inline void       addLine( const Line & line );
        inline void       addLine( const glm::vec3 & start,
                                   const glm::vec3 & end,
                                   const glm::vec3 & color  = glm::vec3 { 1.f },
                                   const float       radius = 0.75f,
                                   const bool        capped = true );
        inline void       addLine( const Sphere &    start,
                                   const Sphere &    end,
                                   const float       radius = 0.75f,
                                   const glm::vec3 & color  = glm::vec3 { 1.f } );

        /* --------------------------- ----- --------------------------- */
        /* --------------------------- PATHS --------------------------- */

        std::vector<Path> paths;
        inline void       addPath( const Path & path );
        inline void       addPath( const rvtx::Path<glm::vec3> & path,
                                   const glm::vec3               pathColor     = glm::vec3 { 1.f },
                                   const float                   pathRadius    = 1.f,
                                   const uint32_t                segmentsCount = 1000 );
        inline void       addPath( const rvtx::Path<glm::vec3> & path,
                                   const glm::vec3               pathColor         = glm::vec3 { 1.f },
                                   const float                   pathRadius        = 1.f,
                                   const glm::vec3               segmentStepColor  = glm::vec3 { 1.f },
                                   const float                   segmentStepRadius = 1.f,
                                   const uint32_t                segmentsCount     = 1000,
                                   const bool                    segmentsVisible   = true );
    };
} // namespace rvtx::gl

#include "rvtx/gl/system/debug_primitives.inl"

#endif // RVTX_GL_SYSTEM_DEBUG_PRIMITIVES_HPP

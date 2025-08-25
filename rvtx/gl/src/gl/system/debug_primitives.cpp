#include "rvtx/gl/system/debug_primitives.hpp"

#include <rvtx/core/logger.hpp>
#include <rvtx/system/path_interpolator.hpp>

namespace rvtx::gl
{
    void DebugPrimitives::update()
    {
        if ( holder == nullptr )
        {
            logger::error( "Attempting to update a null 'DebugPrimitiveHolder'" );
            return;
        }

        std::vector<glm::vec4> nodesPositions;
        std::vector<glm::vec4> nodesColors;
        std::vector<uint32_t>  edgesIndices;
        std::vector<glm::vec4> edgesParams;

        /* --------------------------- ------- --------------------------- */
        /* --------------------------- SPHERES --------------------------- */
        nodesPositions.reserve( spheres.size() );
        nodesColors.reserve( spheres.size() );
        for ( const Sphere & sphere : spheres )
        {
            nodesPositions.emplace_back( sphere.position, sphere.radius );
            nodesColors.emplace_back( sphere.color, sphere.visible ? 1.f : 0.f );
        }

        /* --------------------------- ----- --------------------------- */
        /* --------------------------- LINES --------------------------- */
        nodesPositions.reserve( nodesPositions.size() + lines.size() * 2 );
        nodesColors.reserve( nodesColors.size() + lines.size() * 2 );
        edgesIndices.reserve( lines.size() * 2 );
        edgesParams.reserve( lines.size() );
        uint32_t i = static_cast<uint32_t>( nodesPositions.size() );
        for ( const Line & line : lines )
        {
            nodesPositions.emplace_back( line.start.position, line.start.radius );
            nodesColors.emplace_back( line.start.color, line.capped ? 1.f : 0.f );
            nodesPositions.emplace_back( line.end.position, line.end.radius );
            nodesColors.emplace_back( line.end.color, line.capped ? 1.f : 0.f );

            edgesIndices.emplace_back( i++ );
            edgesIndices.emplace_back( i++ );
            edgesParams.emplace_back( line.color, line.radius );
        }

        /* --------------------------- ----- --------------------------- */
        /* --------------------------- PATHS --------------------------- */
        for ( Path & path : paths )
        {
            PathKeyframeInterpolator<glm::vec3> interpolator {
                &path.path, static_cast<float>( path.segmentsCount ) / path.path.getDuration()
            };
            const uint32_t keyFrameCount  = interpolator.getFrameCount();
            const uint32_t nodeStartIndex = nodesPositions.size();

            nodesPositions.reserve( nodesPositions.size() + keyFrameCount );
            nodesColors.reserve( nodesColors.size() + keyFrameCount );
            edgesIndices.reserve( edgesIndices.size() + keyFrameCount * 2 );
            edgesParams.reserve( edgesParams.size() + keyFrameCount );

            for ( uint32_t i = 0; i < keyFrameCount; i++ )
            {
                nodesPositions.emplace_back( interpolator.valueAt( i ), path.segmentStepRadius );
                nodesColors.emplace_back( path.segmentStepColor, path.segmentsVisible );
            }

            for ( uint32_t i = 0; i < keyFrameCount - 1; i++ )
            {
                edgesIndices.emplace_back( nodeStartIndex + i );
                edgesIndices.emplace_back( nodeStartIndex + i + 1 );
                edgesParams.emplace_back( path.pathColor, path.pathRadius );
            }
        }

        holder->nodesCount = static_cast<uint32_t>( nodesPositions.size() );
        if ( holder->nodesCount > 0 )
        {
            holder->nodesBuffer       = Buffer::Typed<glm::vec4>( nodesPositions );
            holder->nodesColorsBuffer = Buffer::Typed<glm::vec4>( nodesColors );
        }

        holder->edgesCount = static_cast<uint32_t>( edgesIndices.size() );
        if ( holder->edgesCount > 0 )
        {
            holder->edgesBuffer       = Buffer::Typed<uint32_t>( edgesIndices );
            holder->edgesParamsBuffer = Buffer::Typed<glm::vec4>( edgesParams );
        }
    }

    void DebugPrimitives::clear( const bool updateHolder )
    {
        spheres.clear();
        lines.clear();
        paths.clear();

        if ( updateHolder )
            update();
    }
} // namespace rvtx::gl
#include "rvtx/gl/geometry/debug_primitives_geometry.hpp"

#include <GL/gl3w.h>

#include "rvtx/gl/utils/program.hpp"
#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx::gl
{
    void DebugPrimitivesHolder::setNodesColor( const glm::vec4 color )
    {
        nodesColorsBuffer = Buffer::Typed<glm::vec4>( std::vector( nodesCount, color ) );
    }

    void DebugPrimitivesHolder::setNodesColor( const glm::vec3 color, const bool visible )
    {
        setNodesColor( glm::vec4 { color, visible ? 1.f : 0.f } );
    }

    void DebugPrimitivesHolder::setEdgesParams( const glm::vec3 color, const float radius )
    {
        edgesParamsBuffer = Buffer::Typed<glm::vec4>( std::vector( edgesCount / 2, glm::vec4( color, radius ) ) );
    }

    DebugPrimitivesHolder DebugPrimitivesHolder::get( const std::vector<glm::vec4> & nodes,
                                                      std::vector<glm::vec4>         nodesColors,
                                                      const std::vector<uint32_t> &  edgesIndices,
                                                      std::vector<glm::vec4>         edgesParams )
    {
        DebugPrimitivesHolder holder;

        std::size_t colorsCount = nodesColors.size();
        if ( colorsCount < nodes.size() )
        {
            for ( std::size_t i = 0; i < nodes.size(); i++ )
                nodesColors.emplace_back( nodesColors[ i % colorsCount ] );
        }

        std::size_t edgesCount = edgesParams.size();
        if ( edgesCount < edgesIndices.size() / 2 )
        {
            for ( std::size_t i = 0; i < edgesIndices.size() / 2; i++ )
                edgesParams.emplace_back( edgesParams[ i % edgesCount ] );
        }

        holder.nodesCount = static_cast<uint32_t>( nodes.size() );
        if ( holder.nodesCount > 0 )
        {
            holder.nodesBuffer       = Buffer::Typed<glm::vec4>( nodes );
            holder.nodesColorsBuffer = Buffer::Typed<glm::vec4>( nodesColors );
        }

        holder.edgesCount = static_cast<uint32_t>( edgesIndices.size() );
        if ( holder.edgesCount > 0 )
        {
            holder.edgesBuffer       = Buffer::Typed<uint32_t>( edgesIndices );
            holder.edgesParamsBuffer = Buffer::Typed<glm::vec4>( edgesParams );
        }

        return holder;
    }

    DebugPrimitivesHandler::DebugPrimitivesHandler( ProgramManager & manager )
    {
        m_edgesProgram = manager.create( "EdgeGeometry",
                                         { "geometry/debug_primitives/edge/edge.vert",
                                           "geometry/debug_primitives/edge/edge.geom",
                                           "geometry/debug_primitives/edge/edge.frag" } );
        m_nodesProgram = manager.create( "NodeGeometry",
                                         { "geometry/debug_primitives/node/node.vert",
                                           "geometry/debug_primitives/node/node.geom",
                                           "geometry/debug_primitives/node/node.frag" } );

        m_cameraUniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_cameraUniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_cameraUniforms.addValue<bool>( "uIsPerspective", false );

        glCreateVertexArrays( 1, &m_vao );
    }

    DebugPrimitivesHandler::DebugPrimitivesHandler( DebugPrimitivesHandler && other ) noexcept
    {
        std::swap( m_nodesProgram, other.m_nodesProgram );
        std::swap( m_edgesProgram, other.m_edgesProgram );
        std::swap( m_cameraUniforms, other.m_cameraUniforms );
        std::swap( m_vao, other.m_vao );
    }

    DebugPrimitivesHandler & DebugPrimitivesHandler::operator=( DebugPrimitivesHandler && other ) noexcept
    {
        std::swap( m_nodesProgram, other.m_nodesProgram );
        std::swap( m_edgesProgram, other.m_edgesProgram );
        std::swap( m_cameraUniforms, other.m_cameraUniforms );
        std::swap( m_vao, other.m_vao );

        return *this;
    }

    DebugPrimitivesHandler::~DebugPrimitivesHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void DebugPrimitivesHandler::render( const Camera & camera, const Scene & scene )
    {
        const glm::mat4 view       = camera.getViewMatrix();
        const glm::mat4 projection = camera.getProjectionMatrix();

        m_cameraUniforms.updateValue( "uProjMatrix", projection );
        m_cameraUniforms.updateValue( "uIsPerspective", camera.isPerspective() );

        glBindVertexArray( m_vao );

        const auto entities = scene.registry.view<DebugPrimitivesHolder>();
        for ( const auto id : entities )
        {
            entt::basic_handle entity = { scene.registry, id };

            const DebugPrimitivesHolder & holder = entity.get<DebugPrimitivesHolder>();

            m_cameraUniforms.updateValue( "uMVMatrix", view );
            m_cameraUniforms.bind();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            if ( holder.nodesCount > 0 ) // Draw nodes
            {
                m_nodesProgram->use();

                holder.nodesBuffer.bind( 1 );
                holder.nodesColorsBuffer.bind( 2 );

                glDrawArrays( GL_POINTS, 0, holder.nodesCount );
            }

            if ( holder.edgesCount > 0 ) // Draw edges
            {
                m_edgesProgram->use();

                holder.edgesBuffer.bind( 3 );
                holder.edgesParamsBuffer.bind( 4 );

                glDrawArrays( GL_LINES, 0, holder.edgesCount );
            }
        }
        glBindVertexArray( 0 );
    }
} // namespace rvtx::gl

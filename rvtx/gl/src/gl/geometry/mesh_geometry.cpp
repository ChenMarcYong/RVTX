#include "rvtx/gl/geometry/mesh_geometry.hpp"

#include <GL/gl3w.h>

#include "rvtx/gl/utils/program.hpp"
#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx::gl
{
    MeshHolder MeshHolder::get( const Mesh & mesh )
    {
        MeshHolder holder;

        holder.indicesCount = mesh.indices.size();

        holder.indicesBuffer  = Buffer::Typed<uint32_t>( mesh.indices );
        holder.verticesBuffer = Buffer::Typed<Mesh::Vertex>( mesh.vertices );
        holder.idsBuffer      = Buffer::Typed<uint32_t>( mesh.ids );
        holder.useSingleId    = mesh.ids.size() == 1;

        return holder;
    }

    MeshHandler::MeshHandler( ProgramManager & manager )
    {
        m_program = manager.create( "MeshGeometry", { "geometry/mesh/mesh.vert", "geometry/mesh/mesh.frag" } );

        m_uniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_uniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_uniforms.addValue<bool>( "uIsPerspective", true );
        m_uniforms.addValue<bool>( "uUseSingleId", true );

        glCreateVertexArrays( 1, &m_vao );
    }

    MeshHandler::MeshHandler( MeshHandler && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );
    }

    MeshHandler & MeshHandler::operator=( MeshHandler && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_uniforms, other.m_uniforms );

        return *this;
    }

    MeshHandler::~MeshHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void MeshHandler::render( const Camera & camera, const Scene & scene )
    {
        const glm::mat4 view       = camera.getViewMatrix();
        const glm::mat4 projection = camera.getProjectionMatrix();
        m_uniforms.updateValue( "uProjMatrix", projection );
        m_uniforms.updateValue( "uIsPerspective", camera.isPerspective() );
        glBindVertexArray( m_vao );

        const auto entities = scene.registry.view<Transform, MeshHolder>();
        for ( const auto id : entities )
        {
            entt::basic_handle entity          = { scene.registry, id };
            const auto & [ transform, holder ] = entity.get<Transform, MeshHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            m_uniforms.updateValue( "uMVMatrix", view * transform.get() );
            m_uniforms.updateValue( "uUseSingleId", holder.useSingleId );

            m_program->use();

            m_uniforms.bind();
            holder.verticesBuffer.bind( 1 );
            holder.idsBuffer.bind( 2 );

            holder.indicesBuffer.bind();
            glDrawElements( GL_TRIANGLES, holder.indicesCount, GL_UNSIGNED_INT, nullptr );
            holder.indicesBuffer.unbind();
        }
        glBindVertexArray( 0 );
    }
} // namespace rvtx::gl

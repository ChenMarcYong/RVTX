#include "rvtx/gl/geometry/ssesdf_geometry.hpp"

#include <GL/gl3w.h>
#include <glm/mat4x4.hpp>

#include "rvtx/gl/utils/program.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx::gl
{
    SsesdfHolder::SsesdfHolder( const Molecule & molecule, const float probeRadius ) :
        aabb( molecule.getAabb() ), data( molecule.data, { aabb.min, aabb.max }, probeRadius ),
        surface( data.getGraphics() )
    {
        glCreateVertexArrays( 1, &atomVao );
        glBindVertexArray( atomVao );
        glBindBuffer( GL_ARRAY_BUFFER, surface.atoms.handle );
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>( surface.atoms.offset ) );

        glCreateVertexArrays( 1, &sphericalTriangleVao );
        glBindVertexArray( sphericalTriangleVao );

        constexpr GLuint in_pos   = 0;
        constexpr GLuint in_vec_1 = 1;

        glBindBuffer( GL_ARRAY_BUFFER, surface.concavePatchesPosition.handle );
        glEnableVertexAttribArray( in_pos );
        glVertexAttribPointer( in_pos,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.concavePatchesPosition.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, surface.concavePatchesId.handle );
        glEnableVertexAttribArray( in_vec_1 );
        glVertexAttribIPointer(
            in_vec_1, 4, GL_INT, sizeof( glm::ivec4 ), reinterpret_cast<void *>( surface.concavePatchesId.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        glCreateVertexArrays( 1, &torusVao );
        glBindVertexArray( torusVao );

        const GLuint torusObjPos = 0;
        const GLuint inTorusAxis = 1;
        const GLuint inSphere    = 2;

        glBindBuffer( GL_ARRAY_BUFFER, surface.circlePosition.handle );
        glEnableVertexAttribArray( torusObjPos );
        glVertexAttribPointer( torusObjPos,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.circlePosition.offset ) );
        glBindBuffer( GL_ARRAY_BUFFER, surface.circleAxis.handle );
        glEnableVertexAttribArray( inTorusAxis );
        glVertexAttribPointer( inTorusAxis,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.circleAxis.offset ) );
        glBindBuffer( GL_ARRAY_BUFFER, surface.circleVs.handle );
        glEnableVertexAttribArray( inSphere );
        glVertexAttribPointer(
            inSphere, 4, GL_FLOAT, GL_FALSE, sizeof( glm::vec4 ), reinterpret_cast<void *>( surface.circleVs.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glBindVertexArray( 0 );
    }

    SsesdfHolder::SsesdfHolder( SsesdfHolder && other ) noexcept :
        aabb( std::move( aabb ) ), data( std::move( data ) ), surface( std::move( surface ) ),
        atomVao( std::move( atomVao ) ), torusVao( std::move( torusVao ) ),
        sphericalTriangleVao( std::move( sphericalTriangleVao ) )
    {
    }

    SsesdfHolder & SsesdfHolder::operator=( SsesdfHolder && other ) noexcept
    {
        std::swap( aabb, other.aabb );
        std::swap( data, other.data );
        std::swap( surface, other.surface );
        std::swap( atomVao, other.atomVao );
        std::swap( torusVao, other.torusVao );
        std::swap( sphericalTriangleVao, other.sphericalTriangleVao );

        return *this;
    }

    SsesdfHolder::~SsesdfHolder()
    {
        if ( glIsVertexArray( atomVao ) )
            glDeleteVertexArrays( 1, &atomVao );
        if ( glIsVertexArray( torusVao ) )
            glDeleteVertexArrays( 1, &torusVao );
        if ( glIsVertexArray( sphericalTriangleVao ) )
            glDeleteVertexArrays( 1, &sphericalTriangleVao );
    }

    SsesdfHolder SsesdfHolder::get( const Molecule & molecule, float probeRadius ) { return { molecule, probeRadius }; }

    void SsesdfHolder::update() const
    {
        data.build();
        surface = data.getGraphics();

        glBindVertexArray( atomVao );
        glBindBuffer( GL_ARRAY_BUFFER, surface.atoms.handle );
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>( surface.atoms.offset ) );

        glBindVertexArray( sphericalTriangleVao );

        constexpr GLuint in_pos   = 0;
        constexpr GLuint in_vec_1 = 1;

        glBindBuffer( GL_ARRAY_BUFFER, surface.concavePatchesPosition.handle );
        glEnableVertexAttribArray( in_pos );
        glVertexAttribPointer( in_pos,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.concavePatchesPosition.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, surface.concavePatchesId.handle );
        glEnableVertexAttribArray( in_vec_1 );
        glVertexAttribIPointer(
            in_vec_1, 4, GL_INT, sizeof( glm::ivec4 ), reinterpret_cast<void *>( surface.concavePatchesId.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        glBindVertexArray( torusVao );

        const GLuint torusObjPos = 0;
        const GLuint inTorusAxis = 1;
        const GLuint inSphere    = 2;

        glBindBuffer( GL_ARRAY_BUFFER, surface.circlePosition.handle );
        glEnableVertexAttribArray( torusObjPos );
        glVertexAttribPointer( torusObjPos,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.circlePosition.offset ) );
        glBindBuffer( GL_ARRAY_BUFFER, surface.circleAxis.handle );
        glEnableVertexAttribArray( inTorusAxis );
        glVertexAttribPointer( inTorusAxis,
                               4,
                               GL_FLOAT,
                               GL_FALSE,
                               sizeof( glm::vec4 ),
                               reinterpret_cast<void *>( surface.circleAxis.offset ) );
        glBindBuffer( GL_ARRAY_BUFFER, surface.circleVs.handle );
        glEnableVertexAttribArray( inSphere );
        glVertexAttribPointer(
            inSphere, 4, GL_FLOAT, GL_FALSE, sizeof( glm::vec4 ), reinterpret_cast<void *>( surface.circleVs.offset ) );

        glBindBuffer( GL_ARRAY_BUFFER, 0 );
        glBindVertexArray( 0 );
    }

    SsesdfHandler::SsesdfHandler( ProgramManager & manager )
    {
        m_atomProgram              = manager.create( //
            "SsesdfSphereGeometry",
            { "geometry/surface/ssesdf/sphere.vert",
                           "geometry/surface/ssesdf/sphere.geom",
                           "geometry/surface/ssesdf/sphere.frag" } );
        m_sphericalTriangleProgram = manager.create( //
            "SsesdfSphericalTriangleGeometry",
            { "geometry/surface/ssesdf/spherical_triangle.vert",
              "geometry/surface/ssesdf/spherical_triangle.geom",
              "geometry/surface/ssesdf/spherical_triangle.frag" } );
        m_torusProgram             = manager.create( "SsesdfTorusGeometry",
                                                     { "geometry/surface/ssesdf/torus.vert",
                                                       "geometry/surface/ssesdf/torus.geom",
                                                       "geometry/surface/ssesdf/torus.frag" } );

        m_uniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_uniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_uniforms.addValue<float>( "uProbeRadius" );
    }

    void SsesdfHandler::render( const Camera & camera, const Scene & scene )
    {
        const auto entities = scene.registry.view<Molecule, Transform, SsesdfHolder>();

        const glm::mat4 view = camera.getViewMatrix();
        const glm::mat4 proj = camera.getProjectionMatrix();

        m_uniforms.updateValue( "uProjMatrix", proj );

        m_uniforms.bind();

        for ( const auto id : entities )
        {
            entt::basic_handle entity                    = { scene.registry, id };
            const auto & [ molecule, transform, holder ] = entity.get<Molecule, Transform, SsesdfHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            const glm::mat4 mv = view * transform.get();
            m_uniforms.updateValue( "uMVMatrix", mv );
            m_uniforms.updateValue( "uProbeRadius", holder.surface.probeRadius );

            glBindBufferRange( GL_SHADER_STORAGE_BUFFER,
                               1,
                               holder.surface.atoms.handle,
                               holder.surface.atoms.offset,
                               holder.surface.atoms.size );
            glBindBufferRange( GL_SHADER_STORAGE_BUFFER,
                               2,
                               holder.surface.concavePatchesNeighbors.handle,
                               holder.surface.concavePatchesNeighbors.offset,
                               holder.surface.concavePatchesNeighbors.size );

            if ( holder.surface.concavePatchNb > 0 )
            {
                m_sphericalTriangleProgram->use();
                glBindVertexArray( holder.sphericalTriangleVao );
                glDrawArrays( GL_POINTS, 0, holder.surface.concavePatchNb );
                glBindVertexArray( 0 );
            }

            if ( holder.surface.circlePatchNb > 0 )
            {
                m_torusProgram->use();
                glBindVertexArray( holder.torusVao );
                glDrawArrays( GL_POINTS, 0, holder.surface.circlePatchNb );
                glBindVertexArray( 0 );
            }

            if ( holder.surface.atomNb > 0 )
            {
                m_atomProgram->use();
                glBindVertexArray( holder.atomVao );
                glDrawArrays( GL_POINTS, 0, holder.surface.atomNb );
                glBindVertexArray( 0 );
            }
        }
    }
} // namespace rvtx::gl

#include "rvtx/gl/geometry/sesdf_geometry.hpp"

#include <GL/gl3w.h>

#include "rvtx/gl/utils/program.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx::gl
{
    SesdfHolder::SesdfHolder( const Molecule & molecule, float probeRadius ) :
        aabb( molecule.getAabb() ), data( molecule.data, { aabb.min, aabb.max }, probeRadius ),
        surface( data.getGraphics() )
    {
        if ( surface.segmentPatches.size > 0 )
        {
            glCreateVertexArrays( 1, &segmentVao );

            glBindVertexArray( segmentVao );
            glBindBuffer( GL_ARRAY_BUFFER, surface.segmentPatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.segmentPatches.offset;
            glVertexAttribIPointer( 0, 4, GL_UNSIGNED_INT, sizeof( glm::uvec4 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        if ( surface.convexPatches.size > 0 )
        {
            glCreateVertexArrays( 1, &circleVao );
            glBindVertexArray( circleVao );

            glBindBuffer( GL_ARRAY_BUFFER, surface.circlePatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.circlePatches.offset;
            glVertexAttribIPointer( 0, 2, GL_UNSIGNED_INT, sizeof( glm::uvec2 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        if ( surface.convexPatches.size > 0 )
        {
            glCreateVertexArrays( 1, &convexVao );
            glBindVertexArray( convexVao );

            glBindBuffer( GL_ARRAY_BUFFER, surface.convexPatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.convexPatches.offset;
            glVertexAttribIPointer( 0, 2, GL_UNSIGNED_INT, sizeof( glm::uvec2 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        glBindVertexArray( 0 );
    }

    SesdfHolder::SesdfHolder( SesdfHolder && other ) noexcept : data( std::move( other.data ) )
    {
        std::swap( surface, other.surface );
        std::swap( segmentVao, other.segmentVao );
        std::swap( circleVao, other.circleVao );
        std::swap( convexVao, other.convexVao );
    }

    SesdfHolder & SesdfHolder::operator=( SesdfHolder && other ) noexcept
    {
        std::swap( surface, other.surface );
        std::swap( segmentVao, other.segmentVao );
        std::swap( circleVao, other.circleVao );
        std::swap( convexVao, other.convexVao );

        return *this;
    }

    SesdfHolder SesdfHolder::get( const Molecule & molecule, float probeRadius ) { return { molecule, probeRadius }; }

    SesdfHolder::~SesdfHolder()
    {
        if ( glIsVertexArray( segmentVao ) )
            glDeleteVertexArrays( 1, &segmentVao );
        if ( glIsVertexArray( circleVao ) )
            glDeleteVertexArrays( 1, &circleVao );
        if ( glIsVertexArray( convexVao ) )
            glDeleteVertexArrays( 1, &convexVao );
    }

    void SesdfHolder::update() const
    {
        data.build();
        surface = data.getGraphics();

        if ( surface.segmentPatches.size > 0 )
        {
            glBindVertexArray( segmentVao );
            glBindBuffer( GL_ARRAY_BUFFER, surface.segmentPatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.segmentPatches.offset;
            glVertexAttribIPointer( 0, 4, GL_UNSIGNED_INT, sizeof( glm::uvec4 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        if ( surface.convexPatches.size > 0 )
        {
            glBindVertexArray( circleVao );

            glBindBuffer( GL_ARRAY_BUFFER, surface.circlePatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.circlePatches.offset;
            glVertexAttribIPointer( 0, 2, GL_UNSIGNED_INT, sizeof( glm::uvec2 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        if ( surface.convexPatches.size > 0 )
        {
            glBindVertexArray( convexVao );

            glBindBuffer( GL_ARRAY_BUFFER, surface.convexPatches.handle );
            glEnableVertexAttribArray( 0 );

            const std::size_t offset = surface.convexPatches.offset;
            glVertexAttribIPointer( 0, 2, GL_UNSIGNED_INT, sizeof( glm::uvec2 ), reinterpret_cast<void *>( offset ) );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );
        }

        glBindVertexArray( 0 );
    }

    SesdfHandler::SesdfHandler( ProgramManager & manager )
    {
        m_convexPatchProgram  = manager.create( "SesdfConvexPatch",
                                                { "geometry/surface/sesdf/convex_patch.vert",
                                                  "geometry/surface/sesdf/convex_patch.geom",
                                                  "geometry/surface/sesdf/convex_patch.frag" } );
        m_segmentPatchProgram = manager.create( "SesdfSegmentPatch",
                                                { "geometry/surface/sesdf/segment_patch.vert",
                                                  "geometry/surface/sesdf/segment_patch.geom",
                                                  "geometry/surface/sesdf/segment_patch.frag" } );
        m_circlePatchProgram  = manager.create( "SesdfCirclePatch",
                                                { "geometry/surface/sesdf/circle_patch.vert",
                                                  "geometry/surface/sesdf/circle_patch.geom",
                                                  "geometry/surface/sesdf/circle_patch.frag" } );
        m_concavePatchProgram = manager.create( "SesdfConcavePatch",
                                                { "geometry/surface/sesdf/concave_patch.vert",
                                                  "geometry/surface/sesdf/concave_patch.geom",
                                                  "geometry/surface/sesdf/concave_patch.frag" } );
        m_uniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_uniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_uniforms.addValue<glm::mat4>( "uInvMVMatrix" );
        m_uniforms.addValue<float>( "uProbeRadius" );
        m_uniforms.addValue<uint16_t>( "uMaxProbeNeighborNb" );

        glCreateVertexArrays( 1, &m_vao );
    }

    SesdfHandler::SesdfHandler( SesdfHandler && other ) noexcept
    {
        std::swap( m_convexPatchProgram, other.m_convexPatchProgram );
        std::swap( m_segmentPatchProgram, other.m_segmentPatchProgram );
        std::swap( m_circlePatchProgram, other.m_circlePatchProgram );
        std::swap( m_concavePatchProgram, other.m_concavePatchProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );
    }
    SesdfHandler & SesdfHandler::operator=( SesdfHandler && other ) noexcept
    {
        std::swap( m_convexPatchProgram, other.m_convexPatchProgram );
        std::swap( m_segmentPatchProgram, other.m_segmentPatchProgram );
        std::swap( m_circlePatchProgram, other.m_circlePatchProgram );
        std::swap( m_concavePatchProgram, other.m_concavePatchProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );

        return *this;
    }

    SesdfHandler::~SesdfHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void SesdfHandler::render( const Camera & camera, const Scene & scene )
    {
        constexpr auto bindBuffer = []( uint32_t bindingPoint, bcs::HandleSpan<GLuint> buffer )
        {
            if ( buffer.size > 0 )
                glBindBufferRange( GL_SHADER_STORAGE_BUFFER, bindingPoint, buffer.handle, buffer.offset, buffer.size );
        };

        const auto entities = scene.registry.view<Molecule, Transform, SesdfHolder>();

        const glm::mat4 view = camera.getViewMatrix();
        const glm::mat4 proj = camera.getProjectionMatrix();

        m_uniforms.updateValue( "uProjMatrix", proj );
        m_uniforms.bind();

        for ( const auto id : entities )
        {
            entt::basic_handle entity                    = { scene.registry, id };
            const auto & [ molecule, transform, holder ] = entity.get<Molecule, Transform, SesdfHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            const glm::mat4 mv = view * transform.get();
            m_uniforms.updateValue( "uMVMatrix", mv );
            m_uniforms.updateValue( "uInvMVMatrix", glm::inverse( mv ) );
            m_uniforms.updateValue( "uProbeRadius", holder.surface.probeRadius );
            m_uniforms.updateValue( "uMaxProbeNeighborNb", holder.surface.maxConcaveNeighbors );

            bindBuffer( 1, holder.surface.atoms );
            bindBuffer( 2, holder.surface.segmentPatches );
            bindBuffer( 3, holder.surface.concavePatchesPosition );
            bindBuffer( 4, holder.surface.concavePatchesId );
            bindBuffer( 5, holder.surface.concavePatchesNeighbors );
            bindBuffer( 6, holder.surface.sectors );

            if ( holder.surface.concavePatchNb > 0 )
            {
                m_concavePatchProgram->use();
                glBindVertexArray( m_vao );
                glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( holder.surface.concavePatchNb ) );
            }

            if ( holder.surface.circlePatchNb )
            {
                m_circlePatchProgram->use();
                glBindVertexArray( holder.circleVao );
                glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( holder.surface.circlePatchNb ) );
            }

            if ( holder.surface.convexPatchNb > 0 )
            {
                m_convexPatchProgram->use();
                glBindVertexArray( holder.convexVao );
                glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( holder.surface.convexPatchNb ) );
            }

            if ( holder.surface.segmentPatchNb > 0 )
            {
                m_segmentPatchProgram->use();
                glBindVertexArray( holder.segmentVao );
                glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( holder.surface.segmentPatchNb ) );
            }

            glBindVertexArray( 0 );
        }
    }

    SesdfStructureHandler::SesdfStructureHandler( ProgramManager & manager ) : m_mode( Mode::Surface )
    {
        m_lineTetrahedronProgram    = manager.create( "SesdfStructureTetrahedronLine",
                                                      { "geometry/surface/sesdf/structure/tetrahedron.vert",
                                                        "geometry/surface/sesdf/structure/tetrahedron_line.geom",
                                                        "geometry/surface/sesdf/structure/tetrahedron.frag" } );
        m_surfaceTetrahedronProgram = manager.create( "SesdfStructureTetrahedronSurface",
                                                      { "geometry/surface/sesdf/structure/tetrahedron.vert",
                                                        "geometry/surface/sesdf/structure/tetrahedron_surface.geom",
                                                        "geometry/surface/sesdf/structure/tetrahedron.frag" } );
        m_uniforms.addValue<glm::mat4>( "uMVPMatrix" );

        glCreateVertexArrays( 1, &m_vao );
    }

    SesdfStructureHandler::SesdfStructureHandler( SesdfStructureHandler && other ) noexcept
    {
        std::swap( m_lineTetrahedronProgram, other.m_lineTetrahedronProgram );
        std::swap( m_surfaceTetrahedronProgram, other.m_surfaceTetrahedronProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );
    }
    SesdfStructureHandler & SesdfStructureHandler::operator=( SesdfStructureHandler && other ) noexcept
    {
        std::swap( m_lineTetrahedronProgram, other.m_lineTetrahedronProgram );
        std::swap( m_surfaceTetrahedronProgram, other.m_surfaceTetrahedronProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );

        return *this;
    }

    SesdfStructureHandler::~SesdfStructureHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void SesdfStructureHandler::render( const Camera & camera, const Scene & scene )
    {
        constexpr auto bindBuffer = []( uint32_t bindingPoint, bcs::HandleSpan<GLuint> buffer )
        {
            if ( buffer.size > 0 )
                glBindBufferRange( GL_SHADER_STORAGE_BUFFER, bindingPoint, buffer.handle, buffer.offset, buffer.size );
        };

        const auto entities = scene.registry.view<Molecule, Transform, SesdfHolder>();

        const glm::mat4 view = camera.getViewMatrix();
        const glm::mat4 proj = camera.getProjectionMatrix();

        m_uniforms.bind();

        if ( m_mode == Mode::Line )
            m_lineTetrahedronProgram->use();
        else
            m_surfaceTetrahedronProgram->use();

        for ( const auto id : entities )
        {
            const entt::const_handle entity              = { scene.registry, id };
            const auto & [ molecule, transform, holder ] = entity.get<Molecule, Transform, SesdfHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            m_uniforms.updateValue( "uMVPMatrix", proj * view * transform.get() );

            bindBuffer( 1, holder.surface.atoms );
            bindBuffer( 3, holder.surface.concavePatchesPosition );
            bindBuffer( 4, holder.surface.concavePatchesId );

            if ( holder.surface.concavePatchNb > 0 )
            {
                glBindVertexArray( m_vao );

                glEnable( GL_LINE_SMOOTH );
                glLineWidth( 4.f );
                glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( holder.surface.concavePatchNb ) );
            }

            glBindVertexArray( 0 );
        }
    }

} // namespace rvtx::gl

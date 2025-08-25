#include "rvtx/gl/geometry/sas_geometry.hpp"

#include <GL/gl3w.h>

#include "rvtx/gl/utils/program.hpp"
#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/visibility.hpp"

namespace rvtx::gl
{
    struct Sphere
    {
        glm::vec3 position;
        float     radius;
        glm::vec3 color;
        float     visibility = 1.f;
    };

    SasHolder SasHolder::getMolecule( const Molecule & molecule, const MoleculeIDs * moleculeIds,
                                    const float         additionalRadius )
    {
        SasHolder holder;
        holder.size = molecule.residentAtoms.size();

        std::vector<Sphere> atomBuffer {};
        atomBuffer.reserve( holder.size );
        std::vector<uint32_t> idsBuffer;
        idsBuffer.reserve( holder.size );
        for ( std::size_t i = molecule.residentAtoms.start; i < molecule.residentAtoms.end; i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color = getChainColor( chain );
            atomBuffer.emplace_back( Sphere { atomData, atomData.w, color } );
            idsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }

        holder.buffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.idsBuffer = Buffer::Typed<uint32_t>( idsBuffer );
        holder.additionalRadius = additionalRadius;

        return holder;
    }
    SasHolder SasHolder::getNonResident( const Molecule & molecule, const MoleculeIDs * moleculeIds,
                                    const float         additionalRadius )
    {
        SasHolder holder;
        holder.size = molecule.atoms.size() - molecule.residentAtoms.size();

        std::vector<Sphere> atomBuffer {};
        atomBuffer.reserve( holder.size );
        std::vector<uint32_t> idsBuffer;
        idsBuffer.reserve( holder.size );
        for ( std::size_t i = molecule.residentAtoms.end; i < molecule.atoms.size(); i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color  = getChainColor( chain );
            float     radius = atomData.w;
            if ( residue.type == Residue::Type::Ion )
            {
                color  = { 0.f, 1.f, 0.f };
                radius = atomData.w * .3f;
            }
            else if ( residue.type == rvtx::Residue::Type::Ligand )
            {
                color  = { 0.f, 0.f, 1.f };
                radius = atomData.w * .3f;
            }

            atomBuffer.emplace_back( Sphere { atomData, radius, color } );
            idsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }

        holder.buffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.idsBuffer = Buffer::Typed<uint32_t>( idsBuffer );
        holder.additionalRadius = additionalRadius;

        return holder;
    }

    SasHolder SasHolder::getSystem( const Molecule & molecule, const MoleculeIDs * moleculeIds,
                                    const float         additionalRadius )
    {
        SasHolder holder {};
        holder.size = molecule.atoms.size();

        std::vector<Sphere> atomBuffer {};
        atomBuffer.reserve( holder.size );
        std::vector<uint32_t> idsBuffer;
        idsBuffer.reserve( holder.size );
        for ( std::size_t i = 0; i < molecule.atoms.size(); i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color  = getChainColor( chain );
            float     radius = atomData.w;
            if ( residue.type == Residue::Type::Ion )
            {
                color  = { 0.f, 1.f, 0.f };
                radius = atomData.w * .3f;
            }
            else if ( residue.type == Residue::Type::Ligand )
            {
                color  = { 0.f, 0.f, 1.f };
                radius = atomData.w * .3f;
            }

            atomBuffer.emplace_back( Sphere { atomData, radius, color } );
            idsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }

        holder.buffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.idsBuffer = Buffer::Typed<uint32_t>( idsBuffer );
        holder.additionalRadius = additionalRadius;

        return holder;
    }

    SasHandler::SasHandler( ProgramManager & manager )
    {
        m_program = manager.create(
            "SasGeometry",
            { "geometry/sphere/sphere.vert", "geometry/sphere/sphere.geom", "geometry/sphere/sphere.frag" } );

        m_uniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_uniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_uniforms.addValue<float>( "uRadiusAdd", 0.f );
        m_uniforms.addValue<bool>( "uIsPerspective", true );

        glCreateVertexArrays( 1, &m_vao );
    }

    SasHandler::SasHandler( SasHandler && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );
    }

    SasHandler & SasHandler::operator=( SasHandler && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );

        return *this;
    }

    SasHandler::~SasHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void SasHandler::render( const Camera & camera, const Scene & scene )
    {
        m_program->use();

        const glm::mat4 view       = camera.getViewMatrix();
        const glm::mat4 projection = camera.getProjectionMatrix();
        m_uniforms.updateValue( "uProjMatrix", projection );
        m_uniforms.updateValue( "uIsPerspective", camera.isPerspective() );
        glBindVertexArray( m_vao );

        const auto entities = scene.registry.view<Molecule, Transform, SasHolder>();
        for ( const auto id : entities )
        {
            entt::basic_handle entity                    = { scene.registry, id };
            const auto & [ molecule, transform, holder ] = entity.get<Molecule, Transform, SasHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            m_uniforms.updateValue( "uMVMatrix", view * transform.get() );
            m_uniforms.updateValue( "uRadiusAdd", holder.additionalRadius );

            m_uniforms.bind();
            holder.buffer.bind( 1 );
            holder.idsBuffer.bind( 2 );

            glDrawArrays( GL_POINTS, 0, holder.size );
        }
        glBindVertexArray( 0 );
    }

} // namespace rvtx::gl

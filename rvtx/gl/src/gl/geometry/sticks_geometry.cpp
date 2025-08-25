#include "rvtx/gl/geometry/sticks_geometry.hpp"

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

    SticksHolder SticksHolder::getMolecule( const Molecule &    molecule,
                                                        const MoleculeIDs * moleculeIds,
                                                        const float         radius )
    {
        SticksHolder holder;

        // Fill spheres
        holder.sphereSize = molecule.atoms.size();
        std::vector<Sphere> atomBuffer;
        atomBuffer.reserve( holder.sphereSize );
        std::vector<uint32_t> sphereIdsBuffer;
        sphereIdsBuffer.reserve( holder.sphereSize );
        for ( std::size_t i = 0; i < molecule.residentAtoms.size(); i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color = getChainColor( chain );
            atomBuffer.emplace_back( Sphere { atomData, radius, color } );
            sphereIdsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }
        holder.sphereBuffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.sphereIdsBuffer = Buffer::Typed<uint32_t>( sphereIdsBuffer );

        // Fill cylinders
        holder.cylinderSize = molecule.bonds.size() * 2;
        std::vector<uint32_t> bondBuffer;
        bondBuffer.reserve( molecule.bonds.size() * 2 );
        std::vector<uint32_t> bondsIdsBuffer;
        bondsIdsBuffer.reserve( molecule.bonds.size() );
        uint32_t i = 0;
        for ( const auto & [ first, second ] : molecule.bonds )
        {
            bondBuffer.emplace_back( static_cast<uint32_t>( first ) );
            bondBuffer.emplace_back( static_cast<uint32_t>( second ) );

            bondsIdsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->bondIds.start + i++ : 0 );
        }

        holder.cylinderBuffer    = Buffer::Typed<uint32_t>( bondBuffer );
        holder.cylinderIdsBuffer = Buffer::Typed<uint32_t>( bondsIdsBuffer );
        holder.cylinderRadius = radius;

        return holder;
    }

    SticksHolder SticksHolder::getNonResident( const Molecule &    molecule,
                                                           const MoleculeIDs * moleculeIds,
                                                           const float         radius )
    {
        SticksHolder holder;

        // Fill spheres
        holder.sphereSize = molecule.atoms.size() - molecule.residentAtoms.size();
        std::vector<Sphere> atomBuffer;
        atomBuffer.reserve( holder.sphereSize );
        std::vector<uint32_t> sphereIdsBuffer;
        sphereIdsBuffer.reserve( holder.sphereSize );
        for ( std::size_t i = molecule.residentAtoms.end; i < molecule.atoms.size(); i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color = getChainColor( chain );
            if ( residue.type == Residue::Type::Ion )
            {
                color = { 0.f, 1.f, 0.f };
            }
            else if ( residue.type == Residue::Type::Ligand )
            {
                color = { 0.f, 0.f, 1.f };
            }

            atomBuffer.emplace_back( Sphere { atomData, radius, color } );
            sphereIdsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }
        holder.sphereBuffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.sphereIdsBuffer = Buffer::Typed<uint32_t>( sphereIdsBuffer );

        // Fill cylinders (no bonds)
        holder.cylinderSize      = 0;
        holder.cylinderBuffer    = Buffer::Typed<uint32_t>( {} );
        holder.cylinderIdsBuffer = Buffer::Typed<uint32_t>( {} );
        holder.cylinderRadius = radius;

        return holder;
    }

    SticksHolder SticksHolder::getSystem( const Molecule &    molecule,
                                                      const MoleculeIDs * moleculeIds,
                                                      const float         radius )
    {
        SticksHolder holder;

        // Fill spheres
        holder.sphereSize = molecule.atoms.size();
        std::vector<Sphere> atomBuffer;
        atomBuffer.reserve( holder.sphereSize );
        std::vector<uint32_t> sphereIdsBuffer;
        sphereIdsBuffer.reserve( holder.sphereSize );
        for ( std::size_t i = 0; i < molecule.atoms.size(); i++ )
        {
            const Atom &    atom     = molecule.atoms[ i ];
            const glm::vec4 atomData = molecule.data[ i ];
            const Residue & residue  = molecule.residues[ atom.residueId ];
            const Chain &   chain    = molecule.chains[ molecule.residues[ atom.residueId ].chainId ];

            glm::vec3 color = getChainColor( chain );
            if ( residue.type == Residue::Type::Ion )
            {
                color = { 0.f, 1.f, 0.f };
            }
            else if ( residue.type == Residue::Type::Ligand )
            {
                color = { 0.f, 0.f, 1.f };
            }

            atomBuffer.emplace_back( Sphere { atomData, radius, color } );
            sphereIdsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->atomIds.start + i : 0 );
        }
        holder.sphereBuffer    = Buffer::Typed<Sphere>( atomBuffer );
        holder.sphereIdsBuffer = Buffer::Typed<uint32_t>( sphereIdsBuffer );

        // Fill cylinders
        holder.cylinderSize = molecule.bonds.size() * 2;
        std::vector<uint32_t> bondBuffer;
        bondBuffer.reserve( molecule.bonds.size() * 2 );
        std::vector<uint32_t> bondsIdsBuffer;
        bondsIdsBuffer.reserve( molecule.bonds.size() );
        uint32_t i = 0;
        for ( const auto & [ first, second ] : molecule.bonds )
        {
            bondBuffer.emplace_back( static_cast<uint32_t>( first ) );
            bondBuffer.emplace_back( static_cast<uint32_t>( second ) );

            bondsIdsBuffer.emplace_back( moleculeIds != nullptr ? moleculeIds->bondIds.start + i++ : 0 );
        }

        holder.cylinderBuffer    = Buffer::Typed<uint32_t>( bondBuffer );
        holder.cylinderIdsBuffer = Buffer::Typed<uint32_t>( bondsIdsBuffer );
        holder.cylinderRadius = radius;

        return holder;
    }

    SticksHandler::SticksHandler( ProgramManager & manager )
    {
        m_sphereProgram = manager.create(
            "SphereGeometry",
            { "geometry/sphere/sphere.vert", "geometry/sphere/sphere.geom", "geometry/sphere/sphere.frag" } );
        m_cylinderProgram = manager.create( "CylinderGeometry",
                                            { "geometry/cylinder/cylinder.vert",
                                              "geometry/cylinder/cylinder.geom",
                                              "geometry/cylinder/cylinder.frag" } );

        m_uniforms.addValue<glm::mat4>( "uMVMatrix" );
        m_uniforms.addValue<glm::mat4>( "uProjMatrix" );
        m_uniforms.addValue<float>( "uCylRadius", 0.f );
        m_uniforms.addValue<bool>( "uIsPerspective", true );

        glCreateVertexArrays( 1, &m_vao );
    }

    SticksHandler::SticksHandler( SticksHandler && other ) noexcept
    {
        std::swap( m_sphereProgram, other.m_sphereProgram );
        std::swap( m_cylinderProgram, other.m_cylinderProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );
    }

    SticksHandler & SticksHandler::operator=( SticksHandler && other ) noexcept
    {
        std::swap( m_sphereProgram, other.m_sphereProgram );
        std::swap( m_cylinderProgram, other.m_cylinderProgram );
        std::swap( m_uniforms, other.m_uniforms );
        std::swap( m_vao, other.m_vao );

        return *this;
    }

    SticksHandler::~SticksHandler()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
    }

    void SticksHandler::render( const Camera & camera, const Scene & scene )
    {
        const glm::mat4 view       = camera.getViewMatrix();
        const glm::mat4 projection = camera.getProjectionMatrix();
        m_uniforms.updateValue( "uProjMatrix", projection );
        m_uniforms.updateValue( "uIsPerspective", camera.isPerspective() );
        glBindVertexArray( m_vao );

        const auto entities = scene.registry.view<Molecule, Transform, SticksHolder>();
        for ( const auto id : entities )
        {
            entt::basic_handle entity                    = { scene.registry, id };
            const auto & [ molecule, transform, holder ] = entity.get<Molecule, Transform, SticksHolder>();

            if ( entity.all_of<Visibility>() && !entity.get<Visibility>().visible )
                continue;

            m_uniforms.updateValue( "uMVMatrix", view * transform.get() );

            // Draw balls
            m_sphereProgram->use();
            m_uniforms.updateValue( "uCylRadius", holder.sphereAdditionalRadius );
            m_uniforms.bind();
            holder.sphereBuffer.bind( 1 );
            holder.sphereIdsBuffer.bind( 2 );
            glDrawArrays( GL_POINTS, 0, holder.sphereSize );

            // Draw sticks
            m_cylinderProgram->use();
            m_uniforms.updateValue( "uCylRadius", holder.cylinderRadius );
            m_uniforms.bind();
            holder.cylinderBuffer.bind( 3 );
            holder.cylinderIdsBuffer.bind( 4 );
            glDrawArrays( GL_LINES, 0, holder.cylinderSize );
        }
        glBindVertexArray( 0 );
    }

} // namespace rvtx::gl

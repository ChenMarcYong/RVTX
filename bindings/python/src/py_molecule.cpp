#include "pyrvtx/py_molecule.hpp"

#include "pyrvtx/py_scene.hpp"
#include "rvtx/gl/geometry/Sas_geometry.hpp"
#include "rvtx/gl/geometry/sticks_geometry.hpp"
#include "rvtx/system/transform.hpp"

#if RVTX_GL
#include "rvtx/gl/geometry/ball_and_stick_geometry.hpp"
#include "rvtx/gl/geometry/sphere_geometry.hpp"
#if RVTX_CUDA
#include "rvtx/gl/geometry/sesdf_geometry.hpp"
#endif
#endif

namespace rvtx
{
    PyMolecule PyMolecule::load( const std::filesystem::path & path,
                                 const RepresentationType      representation,
                                 PyScene *                     scene )
    {
        PyMolecule pyMolecule = scene->loadMolecule( path, representation );

        pyMolecule.scene = scene;

        return pyMolecule;
    }

    PyMolecule PyMolecule::createProcedural( const ProceduralMoleculeGenerator & generator, PyScene * scene )
    {
        PyMolecule pyMolecule = scene->createMolecule( generator.generate(), RepresentationType::vanDerWaals );

        pyMolecule.scene = scene;

        return pyMolecule;
    }

    PyMolecule::~PyMolecule()
    {
        if ( scene != nullptr && scene->registry.valid( self ) )
        {
            scene->registry.destroy( self );
        }
    }

    PyMolecule::PyMolecule( PyMolecule && other ) noexcept
    {
        id             = std::exchange( other.id, nullptr );
        name           = std::exchange( other.name, nullptr );
        data           = std::exchange( other.data, nullptr );
        atoms          = std::exchange( other.atoms, nullptr );
        bonds          = std::exchange( other.bonds, nullptr );
        residues       = std::exchange( other.residues, nullptr );
        chains         = std::exchange( other.chains, nullptr );
        peptideBonds   = std::exchange( other.peptideBonds, nullptr );
        residentAtoms  = std::exchange( other.residentAtoms, nullptr );
        aabb           = std::exchange( other.aabb, nullptr );
        ids            = std::exchange( other.ids, nullptr );
        transform      = std::exchange( other.transform, nullptr );
        representation = std::exchange( other.representation, nullptr );
        visible        = std::exchange( other.visible, nullptr );
        self           = std::exchange( other.self, entt::handle {} );
        scene          = std::exchange( other.scene, nullptr );
    }

    PyMolecule & PyMolecule::operator=( PyMolecule && other ) noexcept
    {
        std::swap( id, other.id );
        std::swap( name, other.name );
        std::swap( data, other.data );
        std::swap( atoms, other.atoms );
        std::swap( bonds, other.bonds );
        std::swap( residues, other.residues );
        std::swap( chains, other.chains );
        std::swap( peptideBonds, other.peptideBonds );
        std::swap( residentAtoms, other.residentAtoms );
        std::swap( aabb, other.aabb );
        std::swap( ids, other.ids );
        std::swap( transform, other.transform );
        std::swap( representation, other.representation );
        std::swap( visible, other.visible );
        std::swap( self, other.self );
        std::swap( scene, other.scene );

        return *this;
    }

    void PyMolecule::setRepresentation( const RepresentationType newRepresentation ) const
    {
#if RVTX_GL
        switch ( representation->representation )
        {
        case RepresentationType::Ses:
        {
#if RVTX_CUDA
            self.remove<gl::SesdfHolder>();
#else
            rvtx::logger::warning(
                "Tried to switch to SES representation, but CUDA is not available, molecule will not render.." );
#endif
            break;
        }

        case RepresentationType::vanDerWaals:
        {
            self.remove<gl::SphereHolder>();
            break;
        }
        case RepresentationType::BallAndStick:
        {
            self.remove<gl::BallAndStickHolder>();
            break;
        }
        case RepresentationType::Sticks:
        {
            self.remove<gl::SticksHolder>();
            break;
        }
        case RepresentationType::Sas:
        {
            self.remove<gl::SasHolder>();
            break;
        }
        case RepresentationType::Cartoon:
        {
            rvtx::logger::warning( "Cartoon reprensentation is not implemented yet!" );
            break;
        }
        default: break;
        }
#endif
        representation->representation = newRepresentation;
    }

    PyRepresentation::PyRepresentation( const RepresentationType representation ) : representation { representation } {}
} // namespace rvtx

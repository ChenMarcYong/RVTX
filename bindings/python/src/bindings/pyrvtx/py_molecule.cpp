#include "pyrvtx/py_molecule.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"
#include "pyrvtx/py_glm.hpp"
#include "pyrvtx/py_scene.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyMolecule )
    {
        nb::class_<PyMolecule> molecule( m, "Molecule" );

        nb::enum_<RepresentationType>( molecule, "Representation" )
            .value( "vanDerWaals", RepresentationType::vanDerWaals )
            .value( "BallAndStick", RepresentationType::BallAndStick )
            .value( "Ses", RepresentationType::Ses )
            .value( "Sas", RepresentationType::Sas )
            .value( "Sticks", RepresentationType::Sticks )
            .value( "Cartoon", RepresentationType::Cartoon )
            .export_values();

        molecule
            .def_prop_rw(
                "position",
                []( const PyMolecule & m ) { return m.transform->position; },
                []( const PyMolecule & m, const glm::vec3 position ) { m.transform->position = position; },
                nb::rv_policy::copy )
            .def_prop_rw(
                "rotation",
                []( const PyMolecule & m ) { return m.transform->rotation; },
                []( const PyMolecule & m, const glm::quat rotation ) { m.transform->rotation = rotation; },
                nb::rv_policy::copy )
            .def_ro( "id", &PyMolecule::id )
            .def_ro( "name", &PyMolecule::name )
            .def_ro( "data", &PyMolecule::data )
            .def_ro( "atoms", &PyMolecule::atoms )
            .def_ro( "bonds", &PyMolecule::bonds )
            .def_ro( "residues", &PyMolecule::residues )
            .def_ro( "chains", &PyMolecule::chains )
            .def_ro( "peptide_bonds", &PyMolecule::peptideBonds )
            .def_ro( "resident_atoms", &PyMolecule::residentAtoms )
            .def_ro( "aabb", &PyMolecule::aabb )
            .def_rw( "transform", &PyMolecule::transform )
            .def_rw( "T", &PyMolecule::transform )
            .def_prop_rw(
                "representation",
                []( const PyMolecule & m ) { return m.representation->representation; },
                &PyMolecule::setRepresentation );

        m.def( "load_molecule",
               &PyMolecule::load,
               "path"_a,
               "representation"_a = RepresentationType::vanDerWaals,
               "scene"_a          = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );
    }

} // namespace rvtx

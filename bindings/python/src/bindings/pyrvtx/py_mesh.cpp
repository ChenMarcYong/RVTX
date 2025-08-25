#include "pyrvtx/py_mesh.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"
#include "pyrvtx/py_glm.hpp"
#include "pyrvtx/py_scene.hpp"
#include "rvtx/gl/geometry/mesh_geometry.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyMesh )
    {
        nb::class_<Mesh::Vertex>( m, "Vertex" )
            .def( nb::init<>() )
            .def_rw( "position", &Mesh::Vertex::position )
            .def_rw( "normal", &Mesh::Vertex::normal )
            .def_rw( "color", &Mesh::Vertex::color );

        nb::class_<PyMesh>( m, "Mesh" )
            .def_prop_rw(
                "position",
                []( const PyMesh & m ) { return m.transform->position; },
                []( const PyMesh & m, const glm::vec3 position ) { m.transform->position = position; },
                nb::rv_policy::copy )
            .def_prop_rw(
                "rotation",
                []( const PyMesh & m ) { return m.transform->rotation; },
                []( const PyMesh & m, const glm::quat rotation ) { m.transform->rotation = rotation; },
                nb::rv_policy::copy )
            .def_prop_rw(
                "colors",
                []( const PyMesh & pyMesh ) { return pyMesh.getMesh().getColors(); },
                []( PyMesh & pyMesh, const std::vector<glm::vec4> & colors )
                { return pyMesh.getMesh().setColors( colors ); },
                nb::rv_policy::copy )
            .def_prop_ro(
                "positions", []( const PyMesh & pyMesh ) { return pyMesh.getMesh().getPositions(); }, nb::rv_policy::copy )
            .def_prop_ro(
                "normals", []( const PyMesh & pyMesh ) { return pyMesh.getMesh().getNormals(); }, nb::rv_policy::copy )
            .def_ro( "vertices", &PyMesh::vertices, nb::rv_policy::copy )
            .def_ro( "ids", &PyMesh::ids, nb::rv_policy::copy )
            .def_ro( "indices", &PyMesh::indices, nb::rv_policy::copy )
            .def_ro( "aabb", &PyMesh::aabb )
            .def_rw( "transform", &PyMesh::transform )
            .def_rw( "T", &PyMesh::transform )
            .def(
                "compute_charges",
                []( PyMesh &           pyMesh,
                    const PyMolecule & molecule,
                    const bool         updateColors,
                    const float        probeRadius,
                    const float        distance,
                    const bool         useAccelerationStructure )
                {
                    return pyMesh.getMesh().computeCharges(
                        molecule.self.get<Molecule>(), updateColors, probeRadius, distance, useAccelerationStructure );
                },
                "molecule"_a,
                "update_colors"_a              = true,
                "probe_radius"_a               = 1.4f,
                "distance"_a                   = 5.f,
                "use_acceleration_structure"_a = true )
            .def( "update",
                  []( PyMesh & pyMesh )
                  {
                      if ( pyMesh.self.all_of<gl::MeshHolder>() )
                          pyMesh.self.remove<gl::MeshHolder>();
                  } );

        m.def( "load_mesh", &PyMesh::load, "path"_a, "scene"_a = RVTX_PY_MAIN_SCENE, nb::rv_policy::move );
    }
} // namespace rvtx

#include "pyrvtx/py_scene.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT_C( PyScene )
    {
        c.def( nb::init<>() )
            .def_rw( "main_camera", &PyScene::mainCamera )
            .def( "create_camera",
                  &PyScene::createCamera,
                  "transform"_a  = Transform {},
                  "target"_a     = Camera::Target {},
                  "projection"_a = Camera::Projection::Perspective,
                  "controller"_a = CameraController::Type::Trackball,
                  "viewport"_a   = glm::uvec2 { 16, 9 },
                  nb::rv_policy::reference )
            .def(
                "set_main_camera", []( PyScene & scene, PyCamera * camera ) { scene.mainCamera = camera; }, "camera"_a )
            .def( "load_molecule",
                  &PyScene::loadMolecule,
                  "path"_a,
                  "representation"_a = RepresentationType::vanDerWaals,
                  nb::rv_policy::reference )
            .def( "load_mesh", &PyScene::loadMesh, "path"_a, nb::rv_policy::reference )
            .def( "create_molecule", &PyScene::createProceduralMolecule, "generator"_a, nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const std::vector<glm::vec3> &,
                                    const std::vector<float> &>( &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = std::vector { glm::vec3 { 1.f } },
                  "radii"_a  = std::vector { 1.f },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec3> &, const glm::vec3 &, const float>(
                      &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = glm::vec3 { 1.f },
                  "radii"_a  = 1.f,
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const std::vector<float> &,
                                    const std::vector<glm::vec3> &>( &PyScene::createPointCloud ),
                  "points"_a,
                  "radii"_a  = std::vector { 1.f },
                  "colors"_a = std::vector { glm::vec3 { 1.f } },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec3> &, const float, const glm::vec3 &>(
                      &PyScene::createPointCloud ),
                  "points"_a,
                  "radii"_a  = 1.f,
                  "colors"_a = glm::vec3 { 1.f },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec4> &, const std::vector<glm::vec4> &>(
                      &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = std::vector { glm::vec4 { 1.f } },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec4> &, const std::vector<glm::vec3> &>(
                      &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = std::vector { glm::vec3 { 1.f } },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec4> &, const glm::vec4 &>( &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = glm::vec4 { 1.f },
                  nb::rv_policy::reference )
            .def( "create_point_cloud",
                  nb::overload_cast<const std::vector<glm::vec4> &, const glm::vec3 &>( &PyScene::createPointCloud ),
                  "points"_a,
                  "colors"_a = glm::vec3 { 1.f },
                  nb::rv_policy::reference )

            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const std::vector<unsigned> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<glm::vec3> &,
                                    const std::vector<glm::vec3> &>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges"_a,
                  "nodes_radii"_a,
                  "edges_radii"_a,
                  "nodes_colors"_a,
                  "edges_colors"_a,
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const std::vector<unsigned> &,
                                    const float,
                                    const float,
                                    const glm::vec3 &,
                                    const glm::vec3 &>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges"_a,
                  "nodes_radius"_a = 1.f,
                  "edges_radius"_a = 1.f,
                  "nodes_color"_a  = glm::vec3 { 1.f },
                  "edges_color"_a  = glm::vec3 { 1.f },
                  nb::rv_policy::reference )

            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const std::vector<unsigned> &,
                                    const float,
                                    const glm::vec3 &>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges"_a,
                  "edges_radius"_a = 1.f,
                  "edges_color"_a  = glm::vec3 { 1.f },
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec4> &,
                                    const float,
                                    const glm::vec3 &,
                                    const PyGraph::ConnectionType>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges_radius"_a    = 1.f,
                  "edges_color"_a     = glm::vec3 { 1.f },
                  "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const float,
                                    const glm::vec3 &,
                                    const PyGraph::ConnectionType>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges_radius"_a    = 1.f,
                  "edges_color"_a     = glm::vec3 { 1.f },
                  "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec4> &,
                                    const glm::vec3 &,
                                    const float,
                                    const PyGraph::ConnectionType>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges_color"_a     = glm::vec3 { 1.f },
                  "edges_radius"_a    = 1.f,
                  "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const std::vector<glm::vec3> &,
                                    const glm::vec3 &,
                                    const float,
                                    const PyGraph::ConnectionType>( &PyScene::createGraph ),
                  "nodes"_a,
                  "edges_color"_a     = glm::vec3 { 1.f },
                  "edges_radius"_a    = 1.f,
                  "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
                  nb::rv_policy::reference )
            .def( "create_graph",
                  nb::overload_cast<const Path<glm::vec3> &,
                                    const uint32_t,
                                    const glm::vec3 &,
                                    const float,
                                    bool,
                                    const glm::vec3 &,
                                    const float>( &PyScene::createGraph ),
                  "path"_a,
                  "num_points"_a   = 100,
                  "edges_color"_a  = glm::vec3 { 1.f },
                  "edges_radius"_a = 1.f,
                  "show_key"_a     = false,
                  "key_colors"_a   = glm::vec3 { 1.f, 0.f, 0.f },
                  "key_radius"_a   = 1.1f,
                  nb::rv_policy::reference );
    }

} // namespace rvtx

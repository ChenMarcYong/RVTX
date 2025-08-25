#include "pyrvtx/py_graph.hpp"
#include "pyrvtx/py_scene.hpp"

#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyGraphView )
    {
        nb::class_<PyGraphView> graph( m, "Graph" );

        nb::enum_<PyGraph::ConnectionType>( graph, "ConnectionType" )
            .value( "Lines", PyGraph::ConnectionType::LINES )
            .value( "LineStrip", PyGraph::ConnectionType::LINE_STRIP )
            .value( "LineLoop", PyGraph::ConnectionType::LINE_LOOP )
            .export_values();

        graph.def( "update", &PyGraphView::update )
            .def_prop_rw(
                "nodes",
                []( const PyGraphView & g ) { return g.nodes; },
                []( PyGraphView & g, const std::vector<glm::vec4> & nodes )
                {
                    *g.nodes             = nodes;
                    g.graph->needsUpdate = true;
                },
                nb::rv_policy::copy )
            .def_prop_rw(
                "nodes_colors",
                []( const PyGraphView & g ) { return g.nodesColors; },
                []( PyGraphView & g, const std::vector<glm::vec4> & nodesColors )
                {
                    *g.nodesColors       = nodesColors;
                    g.graph->needsUpdate = true;
                },
                nb::rv_policy::copy )
            .def_prop_rw(
                "edges",
                []( const PyGraphView & g ) { return g.edges; },
                []( PyGraphView & g, const std::vector<unsigned int> & edges )
                {
                    *g.edges             = edges;
                    g.graph->needsUpdate = true;
                },
                nb::rv_policy::copy )
            .def_prop_rw(
                "edges_data",
                []( const PyGraphView & g ) { return g.edgesData; },
                []( PyGraphView & g, const std::vector<glm::vec4> & edgesData )
                {
                    *g.edgesData         = edgesData;
                    g.graph->needsUpdate = true;
                },
                nb::rv_policy::copy );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec4> &,
                                 const std::vector<unsigned> &,
                                 const std::vector<glm::vec4> &,
                                 const std::vector<glm::vec4> &,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges"_a,
               "nodes_colors"_a = std::vector { glm::vec4 { 1.f } },
               "edges_data"_a   = std::vector { 1.f },
               "scene"_a        = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const std::vector<unsigned> &,
                                 const std::vector<float> &,
                                 const std::vector<float> &,
                                 const std::vector<glm::vec3> &,
                                 const std::vector<glm::vec3> &,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges"_a,
               "nodes_radii"_a  = std::vector { 1.f },
               "edges_radii"_a  = std::vector { 1.f },
               "nodes_colors"_a = std::vector { glm::vec4 { 1.f } },
               "edges_colors"_a = std::vector { glm::vec4 { 1.f } },
               "_scene"_a       = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const std::vector<unsigned> &,
                                 const float,
                                 const float,
                                 const glm::vec3 &,
                                 const glm::vec3 &,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges"_a,
               "nodes_radius"_a = 1.f,
               "edges_radius"_a = 1.f,
               "nodes_color"_a  = glm::vec3 { 1.f },
               "edges_color"_a  = glm::vec3 { 1.f },
               "scene"_a        = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const std::vector<unsigned> &,
                                 const float,
                                 const glm::vec3 &,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges"_a,
               "edges_radius"_a = 1.f,
               "edges_color"_a  = glm::vec3 { 1.f },
               "scene"_a        = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec4> &,
                                 const float,
                                 const glm::vec3 &,
                                 const PyGraph::ConnectionType,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges_radius"_a    = 1.f,
               "edges_color"_a     = glm::vec3 { 1.f },
               "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
               "scene"_a           = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const float,
                                 const glm::vec3 &,
                                 const PyGraph::ConnectionType,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges_radius"_a    = 1.f,
               "edges_color"_a     = glm::vec3 { 1.f },
               "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
               "scene"_a           = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec4> &,
                                 const glm::vec3 &,
                                 const float,
                                 const PyGraph::ConnectionType,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges_color"_a     = glm::vec3 { 1.f },
               "edges_radius"_a    = 1.f,
               "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
               "scene"_a           = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const glm::vec3 &,
                                 const float,
                                 const PyGraph::ConnectionType,
                                 PyScene &>( &PyGraphView::createGraph ),
               "nodes"_a,
               "edges_color"_a     = glm::vec3 { 1.f },
               "edges_radius"_a    = 1.f,
               "connection_type"_a = PyGraph::ConnectionType::LINE_STRIP,
               "scene"_a           = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_graph",
               nb::overload_cast<const Path<glm::vec3> &,
                                 const uint32_t,
                                 const glm::vec3 &,
                                 const float,
                                 bool,
                                 const glm::vec3 &,
                                 const float,
                                 PyScene &>( &PyGraphView::createGraph ),
               "path"_a,
               "num_points"_a   = 100,
               "edges_color"_a  = glm::vec3 { 1.f },
               "edges_radius"_a = 1.f,
               "show_key"_a     = false,
               "key_colors"_a   = glm::vec3 { 1.f, 0.f, 0.f },
               "key_radius"_a   = 1.1f,
               "scene"_a        = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );
    }

} // namespace rvtx

#include "pyrvtx/py_point_cloud.hpp"
#include "pyrvtx/py_scene.hpp"

#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyPointCloud )
    {
        nb::class_<PyPointCloudView>( m, "PointCloud" )
            .def( "update", &PyPointCloudView::update )
            .def_prop_rw(
                "points",
                []( const PyPointCloudView & pc ) { return pc.points; },
                []( PyPointCloudView & pc, const std::vector<glm::vec4> & points )
                {
                    *pc.points                 = points;
                    pc.pointCloud->needsUpdate = true;
                },
                nb::rv_policy::copy )
            .def_prop_rw(
                "colors",
                []( const PyPointCloudView & pc ) { return pc.pointsColors; },
                []( PyPointCloudView & pc, const std::vector<glm::vec4> & pointsColors )
                {
                    *pc.pointsColors           = pointsColors;
                    pc.pointCloud->needsUpdate = true;
                },
                nb::rv_policy::copy );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const std::vector<glm::vec3> &,
                                 const std::vector<float> &,
                                 PyScene &>( &PyPointCloudView::createPointCloud ),
               "points"_a,
               "colors"_a = std::vector { glm::vec3 { 1.f } },
               "radii"_a  = std::vector { 1.f },
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec3> &, const glm::vec3 &, const float, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "color"_a  = glm::vec3 { 1.f },
               "radius"_a = 1.f,
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec3> &,
                                 const std::vector<float> &,
                                 const std::vector<glm::vec3> &,
                                 PyScene &>( &PyPointCloudView::createPointCloud ),
               "points"_a,
               "radii"_a  = std::vector { 1.f },
               "colors"_a = std::vector { glm::vec3 { 1.f } },
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec3> &, const float, const glm::vec3 &, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "radius"_a = 1.f,
               "color"_a  = glm::vec3 { 1.f },
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec4> &, const std::vector<glm::vec4> &, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "colors"_a = std::vector { glm::vec4 { 1.f } },
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec4> &, const std::vector<glm::vec3> &, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "colors"_a = std::vector { glm::vec3 { 1.f } },
               "scene"_a  = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec4> &, const glm::vec4 &, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "color"_a = glm::vec4 { 1.f },
               "scene"_a = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );

        m.def( "create_point_cloud",
               nb::overload_cast<const std::vector<glm::vec4> &, const glm::vec3 &, PyScene &>(
                   &PyPointCloudView::createPointCloud ),
               "points"_a,
               "color"_a = glm::vec3 { 1.f },
               "scene"_a = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::move );
    }

} // namespace rvtx

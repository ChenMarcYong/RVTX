#include "pyrvtx/py_engine.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <rvtx/gl/window.hpp>

#include "bindings/defines.hpp"
#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyEngine )
    {
        nb::class_<PyEngine>( m, "Engine" )
            .def( nb::init<gl::Window *, const bool, const std::string &>(),
                  "window"_a,
                  "enable_controls"_a   = true,
                  "shaders_root_path"_a = "shaders" )
            .def( nb::init<const uint32_t, const uint32_t, const bool, const std::string &>(),
                  "width"_a             = 1280,
                  "height"_a            = 720,
                  "enable_controls"_a   = true,
                  "shaders_root_path"_a = "shaders" )
            .def_rw( "controls_enabled", &PyEngine::controlsEnabled )
            .def_rw( "enable_controls", &PyEngine::controlsEnabled )
            .def( "render", &PyEngine::render, "scene"_a = RVTX_PY_MAIN_SCENE, "camera"_a = nullptr )
            .def( "update", &PyEngine::update, "scene"_a = RVTX_PY_MAIN_SCENE, "camera"_a = nullptr )
            .def( "get_ids_image", &PyEngine::getIDsImage, nb::rv_policy::copy )
            .def( "get_depth_image", &PyEngine::getDepthImage, nb::rv_policy::copy )
            .def( "get_material_image", &PyEngine::getMaterialImage, nb::rv_policy::copy )
            .def( "get_shading_image", &PyEngine::getShadingImage, nb::rv_policy::copy )
            .def(
                "screenshot",
                []( PyEngine & engine, PyScene & scene, const PyCamera * camera )
                {
                    auto img = engine.screenshot( scene, camera );
                    return nanobind::bytes( reinterpret_cast<const char *>( img.data() ), img.size() );
                },
                "scene"_a  = RVTX_PY_MAIN_SCENE,
                "camera"_a = nullptr,
                nb::rv_policy::copy );
    }
} // namespace rvtx

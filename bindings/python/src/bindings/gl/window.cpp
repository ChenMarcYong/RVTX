#include <rvtx/gl/window.hpp>

#include <nanobind/stl/string.h>

#include "bindings/defines.hpp"

namespace rvtx::gl
{

    RVTX_PY_EXPORT( Window )
    {
        nb::class_<gl::Window>( m, "Window" )
            .def( nb::init<std::string, std::size_t, std::size_t, bool>(),
                  "title"_a  = "rVTX Window",
                  "width"_a  = 1280,
                  "height"_a = 720,
                  "shown"_a  = true );
    }
} // namespace rvtx

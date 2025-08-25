#include "rvtx/system/transform.hpp"
#include "rvtx/core/aabb.hpp"

#include "pyrvtx/py_glm.hpp"
#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( Transform )
    {
        nb::class_<Transform>( m, "Transform" )
            .def( nb::init<const glm::vec3 &, const glm::quat &>(),
                  "position"_a = glm::vec3 { 0.f },
                  "rotation"_a = glm::quat { 1.f, { 0.f, 0.f, 0.f } } )
            .def_rw( "position", &Transform::position, nb::rv_policy::copy )
            .def_rw( "rotation", &Transform::rotation, nb::rv_policy::copy )
            .def( "copy",
                  []( const Transform & t ) {
                      return Transform { t.position, t.rotation };
                  } )
            .def( "set", &Transform::set, "transform"_a )
            .def( "get", &Transform::get, nb::rv_policy::copy )
            .def_prop_ro( "front", &Transform::getFront, nb::rv_policy::copy )
            .def_prop_ro( "left", &Transform::getLeft, nb::rv_policy::copy )
            .def_prop_ro( "up", &Transform::getUp, nb::rv_policy::copy )
            .def( "reset", &Transform::reset )
            .def( "move", &Transform::move, "delta"_a )
            .def( "rotate", nb::overload_cast<const glm::vec3 &>( &Transform::rotate ), "euleur_angles"_a )
            .def( "rotate", nb::overload_cast<const glm::quat &>( &Transform::rotate ), "angles"_a )
            .def( "rotate_pitch", &Transform::rotatePitch, "pitch"_a )
            .def( "rotate_yaw", &Transform::rotateYaw, "yaw"_a )
            .def( "rotate_roll", &Transform::rotateRoll, "roll"_a )
            .def( "look_at", nb::overload_cast<const glm::vec3 &>( &Transform::lookAt ), "point"_a )
            .def( "look_at", nb::overload_cast<const Aabb &, float>( &Transform::lookAt ), "aabb"_a, "fov"_a = 60.f )
            .def( "look_at",
                  nb::overload_cast<const glm::vec3 &, const glm::vec3 &>( &Transform::lookAt ),
                  "front"_a,
                  "up"_a );
    }
} // namespace rvtx

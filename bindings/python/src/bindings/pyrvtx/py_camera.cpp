#include "pyrvtx/py_camera.hpp"

#include "bindings/defines.hpp"
#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PyCamera )
    {
        nb::class_<PyCamera> camera( m, "Camera" );

        nb::class_<Camera::Target>( camera, "Target" )
            .def( nb::init<>() )
            .def( nb::init<const glm::vec3 &, float>(), "position"_a = glm::vec3 { 0.f }, "distance"_a = 20.f )
            .def( nb::init<const AABB &>(), "aabb"_a )
            .def_rw( "position", &Camera::Target::position, nb::rv_policy::copy )
            .def_rw( "distance", &Camera::Target::distance );

        nb::enum_<Camera::Projection>( camera, "Projection" )
            .value( "Orthographic", Camera::Projection::Orthographic )
            .value( "Perspective", Camera::Projection::Perspective )
            .export_values();

        nb::enum_<CameraController::Type>( camera, "Controller" )
            .value( "Trackball", CameraController::Type::Trackball )
            .value( "Freefly", CameraController::Type::Freefly )
            .export_values();

        camera.def_rw( "transform", &PyCamera::transform )
            .def_prop_rw(
                "position",
                []( const PyCamera & c ) { return c.transform->position; },
                []( const PyCamera & c, const glm::vec3 position ) { c.transform->position = position; },
                nb::rv_policy::copy )
            .def_prop_rw(
                "rotation",
                []( const PyCamera & c ) { return c.transform->rotation; },
                []( const PyCamera & c, const glm::quat rotation ) { c.transform->rotation = rotation; },
                nb::rv_policy::copy )
            .def_rw( "T", &PyCamera::transform )
            .def_rw( "viewport", &PyCamera::viewport )
            .def_rw( "fov", &PyCamera::fov )
            .def_rw( "near", &PyCamera::zNear )
            .def_rw( "far", &PyCamera::zFar )
            .def_rw( "target", &PyCamera::target )
            .def_rw( "projection", &PyCamera::projectionType )
            .def_rw( "controller", &PyCamera::controller )
            .def_prop_ro( "is_perspective", []( const PyCamera & c ) { return c.isPerspective(); } )
            .def_prop_ro(
                "current_projection_matrix",
                []( const PyCamera & c ) { return c.getProjectionMatrix(); },
                nb::rv_policy::move )
            .def_prop_ro(
                "orthographic_projection_matrix",
                []( const PyCamera & c ) { return c.getOrthographicProjectionMatrix(); },
                nb::rv_policy::move )
            .def_prop_ro(
                "perspective_projection_matrix",
                []( const PyCamera & c ) { return c.getPerspectiveProjectionMatrix(); },
                nb::rv_policy::move )
            .def_prop_ro( "aspect_ratio", []( const PyCamera & c ) { return c.getAspectRatio(); } )
            .def( "next_projection_type", &Camera::nextProjectionType )
            .def(
                "look_at",
                []( const PyCamera & c, const glm::vec3 & target ) { c.transform->lookAt( target ); },
                "target"_a )
            .def(
                "look_at",
                []( const PyCamera & c, const Aabb & aabb, float fov )
                { c.transform->lookAt( aabb, fov == -1.f ? c.fov : fov ); },
                "aabb"_a,
                "fov"_a = -1.f )
            .def(
                "look_at",
                []( const PyCamera & c, const glm::vec3 & front, const glm::vec3 & up )
                { c.transform->lookAt( front, up ); },
                "front"_a,
                "up"_a )
            .def_prop_ro(
                "front", []( const PyCamera & c ) { return c.transform->getFront(); }, nb::rv_policy::copy )
            .def_prop_ro(
                "left", []( const PyCamera & c ) { return c.transform->getLeft(); }, nb::rv_policy::copy )
            .def_prop_ro(
                "up", []( const PyCamera & c ) { return c.transform->getUp(); }, nb::rv_policy::copy )
            .def(
                "move", []( const PyCamera & c, const glm::vec3 & delta ) { c.transform->move( delta ); }, "delta"_a )
            .def(
                "rotate",
                []( const PyCamera & c, const glm::vec3 & eulerAngles ) { c.transform->rotate( eulerAngles ); },
                "euler_angles"_a )
            .def(
                "rotate",
                []( const PyCamera & c, const glm::quat & angles ) { c.transform->rotate( angles ); },
                "angles"_a )
            .def(
                "rotate_pitch",
                []( const PyCamera & c, const float pitch ) { c.transform->rotatePitch( pitch ); },
                "pitch"_a )
            .def(
                "rotate_yaw", []( const PyCamera & c, const float yaw ) { c.transform->rotateYaw( yaw ); }, "yaw"_a )
            .def(
                "rotate_roll",
                []( const PyCamera & c, const float roll ) { c.transform->rotateRoll( roll ); },
                "roll"_a );

        m.def( "create_camera",
               &PyCamera::create,
               "transform"_a  = Transform {},
               "target"_a     = Camera::Target {},
               "projection"_a = Camera::Projection::Perspective,
               "controller"_a = CameraController::Type::Trackball,
               "viewport"_a   = glm::uvec2 { 16, 9 },
               "scene"_a      = RVTX_PY_MAIN_SCENE,
               nb::rv_policy::reference );

        m.def(
            "set_main_camera",
            []( PyCamera * camera, PyScene & scene ) { scene.mainCamera = camera; },
            "camera"_a,
            "scene"_a = RVTX_PY_MAIN_SCENE );
    }

} // namespace rvtx

#include "rvtx/system/path_interpolator.hpp"

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( PathTimeInterpolator )
    {
        nb::class_<PathTimeInterpolator<glm::vec3>>( m, "Vec3PathTimeInterpolator" )
            .def( nb::init<Path<glm::vec3> *>(), "path"_a )
            .def_prop_ro( "current", &PathTimeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_ro( "current_value", &PathTimeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_ro( "value", &PathTimeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_rw( "current_time",
                          &PathTimeInterpolator<glm::vec3>::currentTime,
                          &PathTimeInterpolator<glm::vec3>::setCurrentTime )
            .def( "reset", &PathTimeInterpolator<glm::vec3>::reset )
            .def_prop_ro( "ended", &PathTimeInterpolator<glm::vec3>::ended )
            .def( "step", &PathTimeInterpolator<glm::vec3>::step );

        nb::class_<PathTimeInterpolator<glm::quat>>( m, "QuatPathTimeInterpolator" )
            .def( nb::init<Path<glm::quat> *>(), "path"_a )
            .def_prop_ro( "current", &PathTimeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_ro( "current_value", &PathTimeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_ro( "value", &PathTimeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_rw( "current_time",
                          &PathTimeInterpolator<glm::quat>::currentTime,
                          &PathTimeInterpolator<glm::quat>::setCurrentTime )
            .def( "reset", &PathTimeInterpolator<glm::quat>::reset )
            .def_prop_ro( "ended", &PathTimeInterpolator<glm::quat>::ended )
            .def( "step", &PathTimeInterpolator<glm::quat>::step );
    }

    RVTX_PY_EXPORT( PathKeyframeInterpolator )
    {
        nb::class_<PathKeyframeInterpolator<glm::vec3>>( m, "Vec3PathKeyframeInterpolator" )
            .def( nb::init<Path<glm::vec3> *, const float>(), "path"_a, "frame_rate"_a = 30 )
            .def_prop_ro( "current", &PathKeyframeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_ro( "current_value", &PathKeyframeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_ro( "value", &PathKeyframeInterpolator<glm::vec3>::current, nb::rv_policy::copy )
            .def_prop_rw( "current_frame",
                          &PathKeyframeInterpolator<glm::vec3>::currentFrame,
                          &PathKeyframeInterpolator<glm::vec3>::setCurrentFrame )
            .def( "reset", &PathKeyframeInterpolator<glm::vec3>::reset )
            .def_prop_ro( "ended", &PathKeyframeInterpolator<glm::vec3>::ended )
            .def( "value_at_frame", &PathKeyframeInterpolator<glm::vec3>::valueAt, nb::rv_policy::copy )
            .def_prop_ro( "frame_count", &PathKeyframeInterpolator<glm::vec3>::getFrameCount )
            .def( "step", &PathKeyframeInterpolator<glm::vec3>::step );

        nb::class_<PathKeyframeInterpolator<glm::quat>>( m, "QuatPathKeyframeInterpolator" )
            .def( nb::init<Path<glm::quat> *, const float>(), "path"_a, "frame_rate"_a = 30 )
            .def_prop_ro( "current", &PathKeyframeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_ro( "current_value", &PathKeyframeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_ro( "value", &PathKeyframeInterpolator<glm::quat>::current, nb::rv_policy::copy )
            .def_prop_rw( "current_frame",
                          &PathKeyframeInterpolator<glm::quat>::currentFrame,
                          &PathKeyframeInterpolator<glm::quat>::setCurrentFrame )
            .def( "reset", &PathKeyframeInterpolator<glm::quat>::reset )
            .def_prop_ro( "ended", &PathKeyframeInterpolator<glm::quat>::ended )
            .def( "value_at_frame", &PathKeyframeInterpolator<glm::quat>::valueAt, nb::rv_policy::copy )
            .def_prop_ro( "frame_count", &PathKeyframeInterpolator<glm::quat>::getFrameCount )
            .def( "step", &PathKeyframeInterpolator<glm::quat>::step );
    }
}

#include "rvtx/core/path.hpp"

#include <nanobind/stl/vector.h>

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( Path )
    {
        nb::class_<Path<glm::quat>>( m, "QuatPath" )
            .def( nb::init<const std::vector<glm::quat> &, float, SplineType>(),
                  "values"_a,
                  "duration"_a    = 1.f,
                  "spline_type"_a = SplineType::CatmullRom )
            .def( "at", &Path<glm::quat>::at )
            .def_prop_rw( "duration", &Path<glm::quat>::getDuration, &Path<glm::quat>::setDuration, "duration"_a )
            .def_prop_rw( "values", &Path<glm::quat>::getValues, &Path<glm::quat>::setValues, "values"_a )
            .def_rw( "spline_type", &Path<glm::quat>::splineType )
            .def( "value_at", &Path<glm::quat>::getValueAt, nb::rv_policy::copy )
            .def( "sample", &Path<glm::quat>::sample, "sample_count"_a, nb::rv_policy::copy );

        nb::class_<Path<glm::vec3>>( m, "Vec3Path" )
            .def( nb::init<const std::vector<glm::vec3> &, float, SplineType>(),
                  "values"_a,
                  "duration"_a    = 1.f,
                  "spline_type"_a = SplineType::CatmullRom )
            .def( "at", &Path<glm::vec3>::at )
            .def_prop_rw( "duration", &Path<glm::vec3>::getDuration, &Path<glm::vec3>::setDuration, "duration"_a )
            .def_prop_rw( "values", &Path<glm::vec3>::getValues, &Path<glm::vec3>::setValues, "values"_a )
            .def_rw( "spline_type", &Path<glm::vec3>::splineType )
            .def( "value_at", &Path<glm::vec3>::getValueAt, nb::rv_policy::copy )
            .def( "sample", &Path<glm::vec3>::sample, "sample_count"_a, nb::rv_policy::copy );
    }
} // namespace rvtx

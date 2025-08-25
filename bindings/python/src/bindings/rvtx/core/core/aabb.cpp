#include "rvtx/core/aabb.hpp"

#include "bindings/defines.hpp"

namespace rvtx
{
    RVTX_PY_EXPORT( AABB )
    {
        // D�finition de la classe AABB avec ses m�thodes et propri�t�s
        nb::class_<AABB>( m, "AABB" )
            .def( nb::init<float, float>(),
                  "min"_a = std::numeric_limits<float>::max(),
                  "max"_a = std::numeric_limits<float>::lowest() )
            .def( nb::init<const glm::vec3 &, const glm::vec3 &>(),
                  "min"_a = glm::vec3 { std::numeric_limits<float>::max() },
                  "max"_a = glm::vec3 { std::numeric_limits<float>::lowest() } )
            .def_rw( "min", &AABB::min )
            .def_rw( "max", &AABB::max )
            .def( "grow", nb::overload_cast<const glm::vec3>( &AABB::grow ), "size"_a )
            .def( "grow", nb::overload_cast<const float>( &AABB::grow ), "size"_a )
            .def( "update", nb::overload_cast<const glm::vec3 &>( &AABB::update ), "point"_a )
            .def( "update", nb::overload_cast<const glm::vec4 &>( &AABB::update ), "sphere"_a )
            .def( "update", nb::overload_cast<const AABB &>( &AABB::update ), "aabb"_a )
            .def_prop_ro( "radius", &AABB::getRadius )
            .def_prop_ro( "invalid", &AABB::isInvalid )
            .def_prop_ro( "centroid", &AABB::getCentroid, nb::rv_policy::copy );
    }
} // namespace rvtx

#ifndef RVTX_CORE_MATH_HPP
#define RVTX_CORE_MATH_HPP

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

namespace rvtx
{
    constexpr float Pi        = 3.14159265359f;
    constexpr float OneOverPi = 1.f / 3.14159265359f;
    constexpr float Tau       = 6.28318530718f;

    template<class Arithmetic1, class Arithmetic2, class Arithmetic3, class Promoted = Arithmetic1>
    constexpr Promoted lerp( Arithmetic1 a, Arithmetic2 b, Arithmetic3 t ) noexcept;

    constexpr std::size_t nextPowerOfTwoValue( const std::size_t baseNumber ) noexcept;
    constexpr std::size_t nextPowerOfTwoExponent( std::size_t baseNumber ) noexcept;

    glm::quat angleBetweenDirs( const glm::vec3 & a, const glm::vec3 & b );
} // namespace rvtx

#include "rvtx/core/math.inl"

#endif // RVTX_CORE_MATH_HPP

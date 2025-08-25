#include <limits>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "rvtx/core/math.hpp"

namespace rvtx
{
    template<class Arithmetic1, class Arithmetic2, class Arithmetic3, class Promoted>
    constexpr Promoted lerp( Arithmetic1 a, Arithmetic2 b, Arithmetic3 t ) noexcept
    {
        return a + t * ( b - a );
    }

    constexpr std::size_t nextPowerOfTwoValue( const std::size_t baseNumber ) noexcept
    {
        std::size_t i = 1;
        while ( baseNumber > i )
            i <<= 1;
        return i;
    }

    constexpr std::size_t nextPowerOfTwoExponent( std::size_t baseNumber ) noexcept
    {
        uint32_t exponent = 0;
        while ( baseNumber >>= 1 )
            exponent++;
        return exponent;
    }
} // namespace rvtx
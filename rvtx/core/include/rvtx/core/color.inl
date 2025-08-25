#include "rvtx/core/color.hpp"

namespace rvtx
{
    constexpr inline float getLuminance( const glm::vec3 x )
    {
        return ( 0.2126f * x.r + 0.7152f * x.g + 0.0722f * x.b );
    }
} // namespace rvtx
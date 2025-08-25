#ifndef RVTX_CORE_RANDOM_HPP
#define RVTX_CORE_RANDOM_HPP

#include <vector>

#include "rvtx/core/math.hpp"

namespace rvtx
{
    inline std::vector<glm::vec3> sphereHalton( int n, int p2 );
} // namespace rvtx

#include "rvtx/core/random.inl"

#endif // RVTX_CORE_RANDOM_HPP

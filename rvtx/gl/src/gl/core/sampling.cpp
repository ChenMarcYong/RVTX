#include "rvtx/gl/core/sampling.hpp"

#include <cmath>

#include <glm/gtc/random.hpp>

#include "rvtx/core/math.hpp"

namespace rvtx::gl
{
    float random( float start, float end ) { return glm::linearRand( start, end ); }

    glm::vec3 cosineWeightedHemisphere( float u, float v )
    {
        const float cosTheta = sqrtf( u );
        const float sinTheta = sqrtf( 1.f - u );
        const float phi      = rvtx::Tau * v;

        return glm::vec3( std::cos( phi ) * sinTheta, std::sin( phi ) * sinTheta, cosTheta );
    }
} // namespace rvtx::gl

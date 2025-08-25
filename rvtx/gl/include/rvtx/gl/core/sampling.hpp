#ifndef RVTX_GL_CORE_SAMPLING_HPP
#define RVTX_GL_CORE_SAMPLING_HPP

#include <glm/vec3.hpp>

namespace rvtx::gl
{
    float random( float start, float end );

    glm::vec3 cosineWeightedHemisphere( float u, float v );
} // namespace rvtx::gl

#endif // RVTX_GL_CORE_SAMPLING_HPP

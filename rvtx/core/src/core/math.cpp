#include "rvtx/core/math.hpp"

#include <glm/gtx/norm.hpp>

namespace rvtx
{
    glm::quat angleBetweenDirs( const glm::vec3 & a, const glm::vec3 & b )
    {
        const float cosTheta = glm::dot( a, b );

        if ( cosTheta < -1.f + 0.001f ) // Opposite case
        {
            glm::vec3 axis = glm::cross( glm::vec3( 0.0f, 0.0f, 1.0f ), a );
            if ( glm::length2( axis ) < 0.01f )
                axis = glm::cross( glm::vec3( 1.0f, 0.0f, 0.0f ), a );

            axis = glm::normalize( axis );
            return glm::angleAxis( glm::pi<float>(), axis );
        }

        const float s    = sqrtf( ( 1.f + cosTheta ) * 2.f );
        const float invs = 1.f / s;

        const glm::vec3 axis = glm::cross( a, b ) * invs;

        return glm::quat { s * 0.5f, axis };
    }
} // namespace rvtx

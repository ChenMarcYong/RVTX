#include "rvtx/system/transform.hpp"

#include "glm/gtx/component_wise.hpp"
#include "rvtx/core/aabb.hpp"

namespace rvtx
{
    Transform::Transform( const glm::vec3 & position, const glm::quat & rotation ) :
        position( position ), rotation( rotation )
    {
    }

    void Transform::lookAt( const glm::vec3 & target )
    {
        // Based on https://stackoverflow.com/a/49824672
        glm::vec3   targetDirection = target - position;
        const float distance        = glm::length( targetDirection );
        if ( distance < 1e-6f )
        {
            rotation = glm::dquat { 1., 0., 0., 0. };
            return;
        }

        targetDirection /= distance;

        const glm::vec3 up       = getUp();
        const float     cosTheta = glm::dot( targetDirection, up );
        if ( glm::abs( glm::abs( cosTheta ) - 1.f ) < 1e-6f )
            rotation = glm::quatLookAt( targetDirection, getFront() );
        else
            rotation = glm::quatLookAt( targetDirection, up );
    }

    void Transform::lookAt( const rvtx::Aabb & aabb, float fov )
    {
        const glm::vec3 target   = ( aabb.min + aabb.max ) * .5f;
        const float     bbRadius = glm::compMax( glm::abs( aabb.max - target ) );
        const float     distance = bbRadius / std::tan( fov * .5f );

        const glm::vec3 direction = getFront();
        position                  = target - direction * 2.f * distance;
        lookAt( target );
    }

    void Transform::lookAt( const glm::vec3 & front, const glm::vec3 & up )
    {
        // Rotate transform to match 'front'
        rotation = angleBetweenDirs( Front, front );

        // Rotate transform's 'up' to match computed 'up'
        const glm::vec3 camUp = getUp();

        const float angleUp = glm::acos( glm::dot( camUp, up ) );

        // If 'up == camUp', skip roll
        if ( std::isfinite( angleUp ) )
            rotateRoll( glm::sign( glm::dot( up, glm::cross( camUp, front ) ) ) * angleUp );
    }
} // namespace rvtx

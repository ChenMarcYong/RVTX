#include "rvtx/system/camera.hpp"

namespace rvtx
{
    Camera::Camera( Transform & transform, glm::uvec2 viewport, Projection type, const float fov ) :
        transform( &transform ), viewport( viewport ), projectionType( type ), fov( fov )
    {
    }

    Camera::Camera( Transform & transform, glm::uvec2 viewport, Target target, Projection type, const float fov ) :
        transform( &transform ), viewport( viewport ), target( target ), projectionType( type ), fov( fov )
    {
    }

    glm::mat4 Camera::getOrthographicProjectionMatrix() const
    {
        const float aspectRatio = getAspectRatio();

        const float top = tanf( fov * 0.5f ) * target.distance;

        const float bottom = -top;
        const float right  = top * aspectRatio;
        const float left   = -top * aspectRatio;

        return glm::ortho( left, right, bottom, top, zNear, zFar );
    }

    glm::mat4 Camera::getPerspectiveProjectionMatrix() const
    {
        return glm::perspective( fov, getAspectRatio(), zNear, zFar );
    }

    Camera::Target::Target( const glm::vec3 & position, float distance ) : position { position }, distance { distance }
    {
    }

    Camera::Target::Target( const Aabb & aabb, const float distanceMultiplicator ) :
        position { aabb.hasAttachedTransform() ? aabb.getTCentroid() : aabb.getCentroid() },
        distance { aabb.getRadius() * distanceMultiplicator }
    {
    }
} // namespace rvtx

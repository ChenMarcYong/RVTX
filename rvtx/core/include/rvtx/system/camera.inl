#include "rvtx/system/camera.hpp"

namespace rvtx
{
    inline glm::mat4 Camera::getViewMatrix() const
    {
        const glm::vec3 position = transform->position;
        return glm::lookAt( position, position + transform->getFront(), transform->getUp() );
    }

    inline glm::mat4 Camera::getProjectionMatrix() const
    {
        switch ( projectionType )
        {
        case Orthographic: return getOrthographicProjectionMatrix();
        case Perspective: return getPerspectiveProjectionMatrix();
        }

        return glm::mat4 {};
    }

    inline bool Camera::isPerspective() const { return projectionType == Projection::Perspective; }

    inline float Camera::getAspectRatio() const
    {
        return static_cast<float>( viewport.x ) / static_cast<float>( viewport.y );
    }

    inline void Camera::nextProjectionType() { projectionType = static_cast<Projection>( !projectionType ); }
} // namespace rvtx

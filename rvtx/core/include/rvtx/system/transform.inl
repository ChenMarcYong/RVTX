#include "rvtx/system/transform.hpp"

namespace rvtx
{
    inline void Transform::set( const glm::mat4 & transform )
    {
        rotation = glm::quat_cast( transform );
        position = transform[ 3 ];
    }

    inline glm::mat4 Transform::get() const
    {
        glm::mat4 transform = glm::mat4_cast( rotation );
        transform[ 3 ]      = glm::vec4( position, 1.f );

        return transform;
    }

    inline void Transform::move( const glm::vec3 & delta )
    {
        position += getLeft() * delta.x;
        position += getUp() * delta.y;
        position += getFront() * delta.z;
    }

    inline void Transform::rotate( const glm::vec3 & angles ) { rotation = rotation * glm::quat( angles ); }

    inline void Transform::rotate( const glm::quat & angles ) { rotation = rotation * angles; }

    inline void Transform::rotatePitch( const float pitch )
    {
        rotation = rotation * glm::quat( glm::vec3( -pitch, 0.0, 0.0 ) );
    }

    inline void Transform::rotateYaw( const float yaw )
    {
        rotation = rotation * glm::quat( glm::vec3( 0.0, -yaw, 0.0 ) );
    }

    inline void Transform::rotateRoll( const float roll )
    {
        rotation = rotation * glm::quat( glm::vec3( 0.0, 0.0, roll ) );
    }

    inline void Transform::rotateAround( const glm::quat & rotationDelta,
                                         const glm::vec3 & target,
                                         const float       distance )
    {
        rotation = rotation * rotationDelta;
        position = rotation * glm::vec3( 0.f, 0.f, distance ) + target;
    }

    inline glm::vec3 Transform::getFront() const { return glm::normalize( glm::mat3_cast( rotation ) * Front ); }

    inline glm::vec3 Transform::getLeft() const { return glm::normalize( glm::mat3_cast( rotation ) * Left ); }

    inline glm::vec3 Transform::getUp() const { return glm::normalize( glm::mat3_cast( rotation ) * Up ); }

    inline void Transform::reset()
    {
        position = glm::vec3 { 0.f };
        rotation = glm::quat { 1.f, { 0.f, 0.f, 0.f } };
    }
} // namespace rvtx
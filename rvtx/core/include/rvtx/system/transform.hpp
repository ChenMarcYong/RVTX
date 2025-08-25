#ifndef RVTX_SYSTEM_TRANSFORM_HPP
#define RVTX_SYSTEM_TRANSFORM_HPP

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "entt/entt.hpp"
#include "rvtx/core/math.hpp"

namespace rvtx
{
    struct Aabb;

    class Transform
    {
      public:
        constexpr static glm::vec3 Front { 0.f, 0.f, -1.f };
        constexpr static glm::vec3 Left { -1.f, 0.f, 0.f };
        constexpr static glm::vec3 Up { 0.f, 1.f, 0.f };

        static constexpr bool in_place_delete = true;

        Transform( const glm::vec3 & position = glm::vec3 { 0.f },
                   const glm::quat & rotation = glm::quat( 1.f, { 0.f, 0.f, 0.f } ) );

        Transform( const Transform & )             = default;
        Transform & operator=( const Transform & ) = default;

        Transform( Transform && ) noexcept             = default;
        Transform & operator=( Transform && ) noexcept = default;

        ~Transform() = default;

        inline void      set( const glm::mat4 & transform );
        inline glm::mat4 get() const;

        void lookAt( const glm::vec3 & target );
        void lookAt( const rvtx::Aabb & aabb, float fov = 60.f );
        void lookAt( const glm::vec3 & front, const glm::vec3 & up );

        inline void move( const glm::vec3 & delta );
        inline void rotate( const glm::vec3 & angles );
        inline void rotate( const glm::quat & angles );
        inline void rotatePitch( const float pitch );
        inline void rotateYaw( const float yaw );
        inline void rotateRoll( const float roll );

        inline void rotateAround( const glm::quat & rotationDelta, const glm::vec3 & target, float distance );

        inline glm::vec3 getFront() const;
        inline glm::vec3 getLeft() const;
        inline glm::vec3 getUp() const;

        inline void reset();

        glm::vec3 position { 0.f };
        glm::quat rotation { 1.f, { 0.f, 0.f, 0.f } };
    };

} // namespace rvtx

#include "rvtx/system/transform.inl"

#endif // RVTX_SYSTEM_TRANSFORM_HPP

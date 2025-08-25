#ifndef RVTX_SYSTEM_CAMERA_HPP
#define RVTX_SYSTEM_CAMERA_HPP

#include "rvtx/system/transform.hpp"
#include "rvtx/core/aabb.hpp"

namespace rvtx
{
    struct Input;
    struct CameraControllerSettings;

    struct Camera
    {
        friend bool trackball( const Input & input, entt::handle entity, const CameraControllerSettings & settings );

        enum Projection : int
        {
            Orthographic = 0,
            Perspective  = 1,
        };

        struct Target
        {
            Target( const glm::vec3 & position = glm::vec3 { 0.f }, float distance = 0.f );
            Target( const Aabb & aabb, const float distanceMultiplicator = 2.f );

            glm::vec3 position {};
            float     distance { 0.f };
        };

        static constexpr bool in_place_delete = true;

        Camera() = default;
        Camera( Transform & transform,
                glm::uvec2  viewport,
                Projection  type = Perspective,
                const float fov  = glm::radians( 45.f ) );
        Camera( Transform & transform,
                glm::uvec2  viewport,
                Target      target = {},
                Projection  type   = Perspective,
                const float fov    = glm::radians( 45.f ) );
        ~Camera() = default;

        glm::mat4 getViewMatrix() const;
        glm::mat4 getProjectionMatrix() const;
        glm::mat4 getOrthographicProjectionMatrix() const;
        glm::mat4 getPerspectiveProjectionMatrix() const;

        inline bool  isPerspective() const;
        inline float getAspectRatio() const;
        inline void  nextProjectionType();

        Transform * transform;
        glm::uvec2  viewport;

        float fov { glm::radians( 45.f ) };
        float zFar { 1e4f };
        float zNear { 1.f };

        Target target {};

        Projection projectionType { Projection::Perspective };

      private:
        glm::vec3 m_velocity {};
        bool      m_needsUpdate { true };
    };
} // namespace rvtx

#include "rvtx/system/camera.inl"

#endif // RVTX_SYSTEM_CAMERA_HPP

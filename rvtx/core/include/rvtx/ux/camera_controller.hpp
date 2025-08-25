#ifndef RVTX_UX_CAMERACONTROLLER_HPP
#define RVTX_UX_CAMERACONTROLLER_HPP

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "rvtx/ux/controller.hpp"

namespace rvtx
{
    struct Input;

    struct CameraControllerSettings
    {
        static constexpr bool in_place_delete = true;

        static const float TranslationSpeedDefault;
        float              translationSpeed = TranslationSpeedDefault;
        static const float TranslationFactorSpeedDefault;
        float              translationFactorSpeed = TranslationFactorSpeedDefault;
        static const float RotationSpeedDefault;
        float              rotationSpeed = RotationSpeedDefault;
        static const float ElasticityFactorDefault;
        float              elasticityFactor = ElasticityFactorDefault;
        static const float ElasticityThresholdDefault;
        float              elasticityThreshold = ElasticityThresholdDefault;
    };
    bool freefly( const rvtx::Input & input, entt::handle entity, const CameraControllerSettings & settings );

    bool trackball( const rvtx::Input & input, entt::handle entity, const CameraControllerSettings & settings );

    class CameraController : public Controller
    {
      public:
        enum class Type
        {
            Freefly,
            Trackball
        };

        CameraController( entt::handle camera );
        ~CameraController() = default;

        void setType( CameraController::Type type );
        bool update( const rvtx::Input & input ) override;
        void setEnabled( const bool state );

        Type type = Type::Trackball;

      private:
        entt::handle             m_camera;
        CameraControllerSettings m_settings {};
        bool                     enabled { true };
    };

} // namespace rvtx

#include "rvtx/ux/camera_controller.inl"

#endif // RVTX_UX_CAMERACONTROLLER_HPP

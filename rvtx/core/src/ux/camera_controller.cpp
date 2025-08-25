#include "rvtx/ux/camera_controller.hpp"

#include <glm/gtx/compatibility.hpp>

#include "rvtx/system/camera.hpp"
#include "rvtx/system/transform.hpp"
#include "rvtx/ux/input.hpp"

namespace rvtx
{
    constexpr float CameraControllerSettings::TranslationSpeedDefault       = 150.f;
    constexpr float CameraControllerSettings::TranslationFactorSpeedDefault = 2.f;
    constexpr float CameraControllerSettings::RotationSpeedDefault          = 5e-3f;
    constexpr float CameraControllerSettings::ElasticityFactorDefault       = 5.f;
    constexpr float CameraControllerSettings::ElasticityThresholdDefault    = 1e-1f;

    bool freefly( const rvtx::Input & input, entt::handle entity, const CameraControllerSettings & settings )
    {
        assert( entity.all_of<Transform>() );

        auto & transform = entity.get<Transform>();
        bool   notify    = false;

        // Rotation.
        if ( input.mouseLeftPressed )
        {
            transform.rotate( glm::vec3( -settings.rotationSpeed * input.deltaMousePosition.y,
                                         -settings.rotationSpeed * input.deltaMousePosition.x,

                                         0.f ) );
            notify = true;
        }
        else if ( input.mouseRightPressed )
        {
            transform.rotateRoll( settings.rotationSpeed * input.deltaMousePosition.x );
            notify = true;
        }

        // Translation.
        glm::vec3 translation {};

        if ( input.isKeyPressed( Key::W ) || input.isKeyPressed( Key::Up ) )
            translation.z++;
        if ( input.isKeyPressed( Key::S ) || input.isKeyPressed( Key::Down ) )
            translation.z--;
        if ( input.isKeyPressed( Key::A ) || input.isKeyPressed( Key::Left ) )
            translation.x++;
        if ( input.isKeyPressed( Key::D ) || input.isKeyPressed( Key::Right ) )
            translation.x--;
        if ( input.isKeyPressed( Key::R ) )
            translation.y++;
        if ( input.isKeyPressed( Key::F ) )
            translation.y--;

        if ( translation != glm::vec3 {} )
        {
            translation *= settings.translationSpeed;
            translation *= input.deltaTime;

            if ( input.isKeyPressed( Key::LShift ) )
                translation *= settings.translationFactorSpeed;
            if ( input.isKeyPressed( Key::LCtrl ) )
                translation /= settings.translationFactorSpeed * 10.f;

            transform.move( translation );
            notify = true;
        }

        return notify;
    }

    bool trackball( const rvtx::Input & input, entt::handle entity, const CameraControllerSettings & settings )
    {
        bool hasComponents = entity.all_of<Transform, Camera>();
        assert( hasComponents && "Entity must have a transform and a camera components." );

        Camera &    camera    = entity.get<Camera>();
        Transform & transform = entity.get<Transform>();

        float deltaDistance = 0.f;
        if ( input.deltaMouseWheel != 0 )
            deltaDistance = static_cast<float>( input.deltaMouseWheel ) * 0.01f;

        // Mouse left.
        glm::vec3 deltaVelocity {};
        if ( input.mouseLeftPressed )
        {
            deltaVelocity.x = -static_cast<float>( input.deltaMousePosition.x ) * 15.f;
            deltaVelocity.y = static_cast<float>( input.deltaMousePosition.y ) * 15.f;
        }
        // Mouse right.
        else if ( input.mouseRightPressed )
        {
            deltaVelocity.z = -static_cast<float>( input.deltaMousePosition.x ) * 15.f;
        }
        // Pan target with wheel button.
        else if ( input.mouseMiddlePressed )
        {
            const float deltaX = static_cast<float>( input.deltaMousePosition.x ) * 0.1f;
            const float deltaY = static_cast<float>( input.deltaMousePosition.y ) * 0.1f;

            const glm::vec3 direction = -glm::vec3( 1.f, 0.f, 0.f ) * deltaX + glm::vec3( 0.f, 1.f, 0.f ) * deltaY;
            camera.target.position += transform.rotation * direction;
            camera.m_needsUpdate = true;
        }

        // Keyboard.
        if ( input.isKeyPressed( Key::W ) || input.isKeyPressed( Key::Up ) )
            deltaDistance = 1.5f * input.deltaTime;
        if ( input.isKeyPressed( Key::S ) || input.isKeyPressed( Key::Down ) )
            deltaDistance = -1.5f * input.deltaTime;
        if ( input.isKeyPressed( Key::A ) || input.isKeyPressed( Key::Left ) )
            deltaVelocity.x = 1e4f * input.deltaTime;
        if ( input.isKeyPressed( Key::D ) || input.isKeyPressed( Key::Right ) )
            deltaVelocity.x = -1e4f * input.deltaTime;
        if ( input.isKeyPressed( Key::R ) )
            deltaVelocity.y = -1e4f * input.deltaTime;
        if ( input.isKeyPressed( Key::F ) )
            deltaVelocity.y = 1e4f * input.deltaTime;
        if ( input.isKeyPressed( Key::Q ) )
            deltaVelocity.z = 1e4f * input.deltaTime;
        if ( input.isKeyPressed( Key::E ) )
            deltaVelocity.z = -1e4f * input.deltaTime;

        // Set values from settings.
        if ( deltaDistance != 0.f )
        {
            deltaDistance *= settings.translationSpeed;

            if ( input.isKeyPressed( Key::LShift ) )
                deltaDistance *= settings.translationFactorSpeed;
            if ( input.isKeyPressed( Key::LCtrl ) )
                deltaDistance /= settings.translationFactorSpeed * 10.f;

            camera.m_needsUpdate = true;
        }

        if ( deltaVelocity != glm::vec3 {} )
        {
            camera.m_velocity.x += settings.rotationSpeed * deltaVelocity.x;
            camera.m_velocity.y += settings.rotationSpeed * deltaVelocity.y;
            camera.m_velocity.z += settings.rotationSpeed * deltaVelocity.z;
        }

        camera.m_needsUpdate |= glm::any( glm::epsilonNotEqual( camera.m_velocity, glm::vec3 { 0.f }, 1e-6f ) );

        bool notify = false;

        // Update if needed.
        if ( camera.m_needsUpdate )
        {
            camera.target.distance = std::clamp( camera.target.distance - deltaDistance, 0.1f, 10000.f );

            const glm::quat rotation
                = { glm::vec3( -camera.m_velocity.y, camera.m_velocity.x, -camera.m_velocity.z ) * input.deltaTime };
            transform.rotateAround( rotation, camera.target.position, camera.target.distance );

            camera.m_needsUpdate = false;
            notify               = true;
        }

        // Handle elasticity.
        if ( camera.m_velocity != glm::vec3 {} )
        {
            camera.m_velocity
                = glm::lerp( camera.m_velocity, glm::vec3 {}, input.deltaTime * settings.elasticityFactor );

            const auto res = glm::lessThan( glm::abs( camera.m_velocity ), glm::vec3( settings.elasticityThreshold ) );
            if ( !input.mouseLeftPressed && glm::all( res ) )
                camera.m_velocity = glm::vec3 {};

            notify = true;
        }

        return notify;
    }

    CameraController::CameraController( entt::handle camera ) : m_camera( camera ) {}

    bool CameraController::update( const rvtx::Input & input )
    {
        if ( enabled )
        {
            switch ( type )
            {
            case Type::Freefly: return freefly( input, m_camera, m_settings ); break;
            case Type::Trackball: return trackball( input, m_camera, m_settings ); break;
            }
        }

        return false;
    }

    void CameraController::setEnabled( const bool state ) { enabled = state; }
} // namespace rvtx

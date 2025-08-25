#include "pyrvtx/py_camera.hpp"

#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    PyCamera PyCamera::create( const Transform &            transform,
                               const Target                 target,
                               const Projection             projectionType,
                               const CameraController::Type controller,
                               const glm::uvec2             viewport,
                               PyScene *                    scene )
    {
        PyCamera camera = scene->createCamera( transform, target, projectionType, controller, viewport );

        camera.scene = scene;

        return camera;
    }

    PyCamera::~PyCamera()
    {
        if ( scene != nullptr && scene->registry.valid( self ) )
        {
            scene->registry.destroy( self );
        }
    }

    PyCamera::PyCamera( PyCamera && other ) noexcept : Camera( other )
    {
        self                = std::exchange( other.self, entt::handle {} );
        scene               = std::exchange( other.scene, nullptr );
        controller          = std::exchange( other.controller, CameraController::Type::Trackball );
        cameraController    = std::exchange( other.cameraController, nullptr );
        controllerForwarder = std::exchange( other.controllerForwarder, nullptr );
    }

    PyCamera & PyCamera::operator=( PyCamera && other ) noexcept
    {
        Camera::operator=( other );

        std::swap( self, other.self );
        std::swap( scene, other.scene );
        std::swap( controller, other.controller );
        std::swap( cameraController, other.cameraController );
        std::swap( controllerForwarder, other.controllerForwarder );

        return *this;
    }
} // namespace rvtx

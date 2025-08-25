#ifndef PYRVTX_PY_CAMERA_HPP
#define PYRVTX_PY_CAMERA_HPP

#include "rvtx/system/camera.hpp"
#include "rvtx/system/transform.hpp"
#include "rvtx/ux/camera_controller.hpp"

namespace rvtx
{
    class PyScene;

    struct PyCamera : Camera
    {
        PyCamera() = default;
        ~PyCamera();

        PyCamera( const PyCamera & )             = delete;
        PyCamera & operator=( const PyCamera & ) = delete;

        PyCamera( PyCamera && other ) noexcept;
        PyCamera & operator=( PyCamera && other ) noexcept;

        static PyCamera create( const Transform &            transform,
                                const Target                 target,
                                const Projection             projectionType,
                                const CameraController::Type controller,
                                const glm::uvec2             viewport,
                                PyScene *                    scene );

        entt::handle self {};
        PyScene *    scene { nullptr };

        CameraController::Type controller = CameraController::Type::Trackball;

        CameraController *                   cameraController;
        std::unique_ptr<ControllerForwarder> controllerForwarder;
    };
} // namespace rvtx

#endif
#ifndef RVTX_DIL_GEOMETRY_HANDLER_HPP
#define RVTX_DIL_GEOMETRY_HANDLER_HPP

#include <memory>
#include <vector>

#include "Graphics/GraphicsEngine/interface/DeviceContext.h"

#include "FirstPersonCamera.hpp"

#include "rvtx/system/scene.hpp"
#include <rvtx/system/camera.hpp>
#pragma once

namespace rvtx::dil
{
    struct GeometryHandler
    {
        virtual void render(const rvtx::Camera& m_Camera, const Scene& scene, Diligent::IDeviceContext* ctx) = 0;

        virtual void BindBuffers(){}
        virtual void setSphereData() {}
        virtual void CreateSphereBuffers() {}
        virtual void initializePSO(){}

        virtual ~GeometryHandler() = default;
    };


    class GeometryForwarder : public GeometryHandler
    {
    public:
        template<typename T, typename... Args>
        T& add(Args&&... args)
        {
            static_assert(std::is_base_of<GeometryHandler, T>::value,
                "T must inherit from GeometryHandler");

            auto obj = std::make_unique<T>(std::forward<Args>(args)...);
            T* ptr = obj.get();
            m_handlers.emplace_back(std::move(obj));
            return *ptr;
        }


        void render(const rvtx::Camera& camera, const Scene& scene, Diligent::IDeviceContext* ctx) override
        {
            for (auto& handler : m_handlers)
                handler->render(camera, scene, ctx);
        }

        void BindBuffers() override
        {
            for (auto& handler : m_handlers)
                handler->BindBuffers();
        }

        void setSphereData() override
        {
            for (auto& handler : m_handlers)
                handler->setSphereData();
        }

        void CreateSphereBuffers() override
        {
            for (auto& handler : m_handlers)
                handler->CreateSphereBuffers();
        }

        void initializePSO() override
        {
            for (auto& handler : m_handlers)
                handler->initializePSO();
        }

    private:
        std::vector<std::unique_ptr<GeometryHandler>> m_handlers;
    };
}


#endif // RVTX_DIL_GEOMETRY_HANDLER_HPP

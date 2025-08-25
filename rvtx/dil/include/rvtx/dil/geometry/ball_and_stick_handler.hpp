#ifndef RVTX_DIL_GEOMETRY_BALL_AND_HANDLER_HPP
#define RVTX_DIL_GEOMETRY_BALL_AND_HANDLER_HPP

#pragma once

#include "BasicMath.hpp"
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "SwapChain.h"
#include "EngineFactory.h"

#include "rvtx/dil/geometry/handler.hpp"
#include "rvtx/dil/geometry/sphere_holder.hpp"
#include "rvtx/dil/utils/pipeline_manager.hpp"

#include "rvtx/system/scene.hpp"
#include <rvtx/system/camera.hpp>

#include "rvtx/dil/geometry/types.hpp"

namespace rvtx::dil
{

    class BallAndStickHandler : public GeometryHandler
    {
    public:


        BallAndStickHandler(PipelineManager& pipelineManager,
            Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory);


        void Render(const Diligent::float4x4& viewMatrix,
            const Diligent::float4x4& projMatrix);


        void render(const Diligent::FirstPersonCamera& m_Camera, const Scene& scene) override;

        void render2(const rvtx::Camera& camera, const rvtx::Scene& scene) override;

        void render_context(const rvtx::Camera& m_Camera, const Scene& scene, Diligent::IDeviceContext* ctx) override;

        void BindBuffers() override;
        void initializePSO() override;


        PipelineManager::PipelineEntry* getPipelineEntry() {return pipelineSphereEntry;};

        Diligent::ShaderResourceVariableDesc VarsSphere[2] =
        {
            { Diligent::SHADER_TYPE_VERTEX, "spheres", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
            { Diligent::SHADER_TYPE_VERTEX, "ids",     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC }
        };
    private:
        // Création des ressources internes
        void finalizePSO();
        void RenderPE(const Diligent::float4x4& viewMatrix,
            const Diligent::float4x4& projMatrix);
        // Structures alignées avec les shaders



    private:

        // ==========================================================
        // SPHERE
        // ==========================================================

        // Pipeline et bindings
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_pPSO_Sphere;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_pSRB_Sphere;

        // Buffers Sphere
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSphereSettingsCB;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSpheresBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pIDsBuffer;              // Nouveau : buffer des IDs
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresBufferView;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresIdsBufferView;   // Nouveau : view des IDs

        // ==========================================================
        // CYLINDER
        // ==========================================================

        // Pipeline et bindings
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_pPSO_Cylinder;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_pSRB_Cylinder;

        // Buffers cylinder
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pCylinderSettingsCB;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pCylindersBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pIDsCylinderBuffer;              // Nouveau : buffer des IDs
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pCylindersBufferView;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pCylindersIdsBufferView;   // Nouveau : view des IDs

        // ==========================================================
        // ELSE
        // ==========================================================

        Diligent::GraphicsPipelineStateCreateInfo PSOStateCreateInfo;

        PipelineManager::PipelineEntry* pipelineSphereEntry;
        PipelineManager::PipelineEntry* pipelineCylinderEntry;

        // ==========================================================
        // RENDU
        // ==========================================================

        Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  m_pDevice;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_pImmediateContext;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>     m_pSwapChain;
        Diligent::RefCntAutoPtr<Diligent::IEngineFactory> m_pEngineFactory;




    };

} // namespace rvtx::dil

#endif
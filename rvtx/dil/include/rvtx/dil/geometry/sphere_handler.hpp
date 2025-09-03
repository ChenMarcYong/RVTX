#ifndef RVTX_DIL_GEOMETRY_SPHERE_HANDLER_HPP
#define RVTX_DIL_GEOMETRY_SPHERE_HANDLER_HPP

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

    class SphereHandler : public GeometryHandler
    {
    public:



        SphereHandler(PipelineManager& pipelineManager,
            Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory);


        // Rendu des sphères (appelé depuis le SampleBase Render)




        void render(const rvtx::Camera& m_Camera, const Scene& scene, Diligent::IDeviceContext* ctx) override;

        void SetSphereBuffers(const SphereHolder& holder);

        void BindBuffers() override;
        void setSphereData() override;
        void CreateSphereBuffers() override;
        void initializePSO() override;

        Diligent::Uint32 GetSphereCount() const { return m_NumSpheres; }



        PipelineManager::PipelineEntry* getPipelineEntry();


        Diligent::ShaderResourceVariableDesc Vars[7] =
        {

            { Diligent::SHADER_TYPE_VERTEX,   "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
            { Diligent::SHADER_TYPE_GEOMETRY, "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
            { Diligent::SHADER_TYPE_PIXEL,    "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },


            { Diligent::SHADER_TYPE_VERTEX,   "spheres", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
            { Diligent::SHADER_TYPE_VERTEX,   "ids",     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
            { Diligent::SHADER_TYPE_GEOMETRY, "spheres", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
            { Diligent::SHADER_TYPE_GEOMETRY, "ids",     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC }
        };


    private:
        // Création des ressources internes
        void finalizePSO();
        void RenderPE(const Diligent::float4x4& viewMatrix,
            const Diligent::float4x4& projMatrix);
        // Structures alignées avec les shaders

        

    private:
        // Pipeline et bindings
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_pPSO_Sphere;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_pSRB_Sphere;

        // Buffers
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSphereSettingsCB;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSpheresBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pIDsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresBufferView;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresIdsBufferView;

        Diligent::GraphicsPipelineStateCreateInfo PSOStateCreateInfo;
        Diligent::ShaderCreateInfo ShaderCreateI;

        // Données
        Scene *scene = nullptr;
        const Sphere* Spheres;
        Diligent::Uint32 m_NumSpheres = 0;

        Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  m_pDevice;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_pImmediateContext;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>     m_pSwapChain;
        Diligent::RefCntAutoPtr<Diligent::IEngineFactory> m_pEngineFactory;

        PipelineManager::PipelineEntry* pipelineEntry;
        bool useGeom = false;

    };

} // namespace rvtx::dil

#endif

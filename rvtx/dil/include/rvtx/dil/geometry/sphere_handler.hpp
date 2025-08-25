#ifndef RVTX_DIL_GEOMETRY_SPHERE_HANDLER_2_HPP
#define RVTX_DIL_GEOMETRY_SPHERE_HANDLER_2_HPP

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

        SphereHandler(Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory,
            const Sphere* spheres,
            Diligent::Uint32 sphereCount, Scene& m_scene);


        SphereHandler(PipelineManager& pipelineManager,
            Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory);


        SphereHandler(PipelineManager &pipelineManager, PipelineData &pipelineData, const Sphere* spheres, Diligent::Uint32 sphereCount, Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory);

        // Initialise la factory avec les objets globaux fournis par SampleBase

       void load();

        // Rendu des sphères (appelé depuis le SampleBase Render)
        void Render(const Diligent::float4x4& viewMatrix,
            const Diligent::float4x4& projMatrix);


        void render(const Diligent::FirstPersonCamera& m_Camera, const Scene& scene) override;

        void render2(const rvtx::Camera& camera, const rvtx::Scene& scene) override;

        void render_context(const rvtx::Camera& m_Camera, const Scene& scene, Diligent::IDeviceContext* ctx) override;
        void SetSphereBuffers(const SphereHolder2& holder);

        void BindBuffers() override;
        void setSphereData() override;
        void CreateSphereBuffers() override;
        void initializePSO() override;

        // Nombre de sphères (utile pour debug ou UI)
        Diligent::Uint32 GetSphereCount() const { return m_NumSpheres; }



        PipelineManager::PipelineEntry* getPipelineEntry();
        
        // Debug
        void helloWorld();
        //Diligent::ShaderResourceVariableDesc Vars[3] =
        //{ 
        //    { Diligent::SHADER_TYPE_VERTEX, "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
        //    { Diligent::SHADER_TYPE_VERTEX, "spheres", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC },
        //    { Diligent::SHADER_TYPE_VERTEX, "ids",     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC }
        //};


        Diligent::ShaderResourceVariableDesc Vars[7] =
        {
            // Constant buffer (b0) : STATIC pour que ce soit copié dans chaque SRB à la création
            { Diligent::SHADER_TYPE_VERTEX,   "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
            { Diligent::SHADER_TYPE_GEOMETRY, "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
            { Diligent::SHADER_TYPE_PIXEL,    "SphereSettings", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC },

            // SRVs (t0, t1) : DYNAMIC/MUTABLE => on les fixe via le SRB avant chaque draw
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pIDsBuffer;              // Nouveau : buffer des IDs
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresBufferView;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresIdsBufferView;   // Nouveau : view des IDs

        Diligent::GraphicsPipelineStateCreateInfo PSOStateCreateInfo;
        Diligent::ShaderCreateInfo ShaderCreateI;

        // Données
        Scene *scene = nullptr;
        const Sphere* Spheres;
        Diligent::Uint32 m_NumSpheres = 0;

        


        // Références vers les objets globaux (non possédés)
        Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  m_pDevice;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_pImmediateContext;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>     m_pSwapChain;
        Diligent::RefCntAutoPtr<Diligent::IEngineFactory> m_pEngineFactory;

        PipelineManager::PipelineEntry* pipelineEntry;
        std::unique_ptr<PipelineManager> pipeline;
        PipelineData pipelineData;

      
        bool useGeom = false;

    };

} // namespace rvtx::dil

#endif

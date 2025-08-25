#pragma once

#include "rvtx/dil/geometry/sphere_handler.hpp"
#include "rvtx/dil/geometry/sphere_holder.hpp"
#include "MapHelper.hpp"

#include "GraphicsUtilities.h"
#include "ColorConversion.h"

#include <windows.h>
#include <string>

#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <cstring> 

using namespace Diligent;




namespace rvtx::dil
{


    SphereHandler::SphereHandler(PipelineManager& pipelineManager,
        RefCntAutoPtr<IRenderDevice> device,
        RefCntAutoPtr<IDeviceContext> context,
        RefCntAutoPtr<ISwapChain> swapChain,
        RefCntAutoPtr<IEngineFactory> engineFactory) :
        m_pDevice(device), m_pImmediateContext(context), m_pSwapChain(swapChain), m_pEngineFactory(engineFactory)
    {
        initializePSO();

        if(useGeom)
        pipelineEntry = pipelineManager.create2("pipeline sphere", { "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.vsh",
            

            "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.gsh" ,
            "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.psh" }, PSOStateCreateInfo, Vars, _countof(Vars));

        else
            pipelineEntry = pipelineManager.create2("pipeline sphere", { "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphereNogs/sphere.vsh",
        "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphereNogs/sphere.psh" }, PSOStateCreateInfo, Vars, _countof(Vars));

        BindBuffers();
    }


    SphereHandler::SphereHandler(PipelineManager& pipelineManager, PipelineData &_pipelineData, const Sphere* spheres, Uint32 sphereCount, RefCntAutoPtr<IRenderDevice> device,
        RefCntAutoPtr<IDeviceContext> context,
        RefCntAutoPtr<ISwapChain> swapChain,
        RefCntAutoPtr<IEngineFactory> engineFactory) :
        Spheres(spheres),
        m_NumSpheres(sphereCount),
        pipelineData(_pipelineData), m_pDevice(device), m_pImmediateContext(context), m_pSwapChain(swapChain), m_pEngineFactory(engineFactory)
    {

        initializePSO();
        auto result = pipelineManager.create("pipeline sphere", { "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.vsh",


            "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.gsh" ,
            "C:/M2 ISICG/Projet M2/rVTX3/rVTX/rvtx/dil/shaders_hlsl/sphere/sphere.psh" }, PSOStateCreateInfo, Vars, _countof(Vars));


        pipelineEntry = result.entry;




        BindBuffers();
        setSphereData();
        CreateSphereBuffers();


    }

    // ==========================================================
    // PIPELINE MANAGER
    // ==========================================================

    PipelineManager::PipelineEntry* SphereHandler::getPipelineEntry()
    {
        return pipelineEntry;
    }


    void SphereHandler::initializePSO()
    {
        PSOStateCreateInfo.PSODesc.Name = "Sphere Impostor Pipeline";
        PSOStateCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

        const auto scDesc = m_pSwapChain->GetDesc();
        // --- Sorties du GBuffer ---
        PSOStateCreateInfo.GraphicsPipeline.NumRenderTargets = 3;
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[0] = TEX_FORMAT_RGBA32_UINT;   // positions+normales packées
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[1] = TEX_FORMAT_RGBA16_FLOAT;  // couleur/matériaux
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[2] = TEX_FORMAT_R32_UINT;      // IDs
        PSOStateCreateInfo.GraphicsPipeline.DSVFormat = TEX_FORMAT_D32_FLOAT;


        PSOStateCreateInfo.GraphicsPipeline.PrimitiveTopology = useGeom
            ? PRIMITIVE_TOPOLOGY_POINT_LIST
            : PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

        // Rasterizer / Depth
        PSOStateCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = True;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthFunc = COMPARISON_FUNC_LESS;
    }

    void SphereHandler::finalizePSO()
    {
        BufferDesc CBDesc;
        CBDesc.Name = "SphereSettings CB";
        CBDesc.Size = sizeof(SphereSettings);
        CBDesc.Usage = USAGE_DYNAMIC;
        CBDesc.BindFlags = BIND_UNIFORM_BUFFER;
        CBDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
        m_pDevice->CreateBuffer(CBDesc, nullptr, &m_pSphereSettingsCB);
        // Associer le constant buffer aux shaders
        m_pPSO_Sphere->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SphereSettings")->Set(m_pSphereSettingsCB);
        m_pPSO_Sphere->GetStaticVariableByName(SHADER_TYPE_GEOMETRY, "SphereSettings")->Set(m_pSphereSettingsCB);
        m_pPSO_Sphere->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SphereSettings")->Set(m_pSphereSettingsCB);

        // Créer le Shader Resource Binding
        m_pPSO_Sphere->CreateShaderResourceBinding(&m_pSRB_Sphere, true);
    }
    // ==========================================================
    // SPHERE FACTORY
    // ==========================================================

    void SphereHandler::setSphereData()
    {


        

        // --- Buffer des sphères ---
        BufferDesc sphereBufferDesc;
        sphereBufferDesc.Name = "Sphere Data Buffer";
        sphereBufferDesc.Size = sizeof(Sphere) * m_NumSpheres;
        sphereBufferDesc.Usage = USAGE_DYNAMIC;
        sphereBufferDesc.BindFlags = BIND_SHADER_RESOURCE;
        sphereBufferDesc.Mode = BUFFER_MODE_STRUCTURED;
        sphereBufferDesc.ElementByteStride = sizeof(Sphere);
        sphereBufferDesc.CPUAccessFlags = CPU_ACCESS_WRITE;

        m_pDevice->CreateBuffer(sphereBufferDesc, nullptr, &m_pSpheresBuffer);

        {
            MapHelper<Sphere> sphereMap(m_pImmediateContext, m_pSpheresBuffer, MAP_WRITE, MAP_FLAG_DISCARD);
            memcpy(sphereMap, Spheres, sizeof(Sphere) * m_NumSpheres);
        }

        BufferViewDesc sphereViewDesc;
        sphereViewDesc.ViewType = BUFFER_VIEW_SHADER_RESOURCE;
        sphereViewDesc.Name = "Sphere Data Buffer SRV";
        sphereViewDesc.ByteOffset = 0;
        sphereViewDesc.ByteWidth = sizeof(Sphere) * m_NumSpheres;
        m_pSpheresBuffer->CreateView(sphereViewDesc, &m_pSpheresBufferView);
    }

    // ==========================================================
    // PIPELINE
    // ==========================================================

    void SphereHandler::CreateSphereBuffers()
    {

        // --- Buffer des IDs ---
        constexpr Uint32 IdsData[] = { 4 };

        BufferDesc idsBufferDesc;
        idsBufferDesc.Name = "Sphere IDs Buffer";
        idsBufferDesc.Size = sizeof(IdsData);
        idsBufferDesc.Usage = USAGE_DYNAMIC;
        idsBufferDesc.BindFlags = BIND_SHADER_RESOURCE;
        idsBufferDesc.Mode = BUFFER_MODE_STRUCTURED;
        idsBufferDesc.ElementByteStride = sizeof(Uint32);
        idsBufferDesc.CPUAccessFlags = CPU_ACCESS_WRITE;

        m_pDevice->CreateBuffer(idsBufferDesc, nullptr, &m_pIDsBuffer);

        {
            MapHelper<Uint32> idsMap(m_pImmediateContext, m_pIDsBuffer, MAP_WRITE, MAP_FLAG_DISCARD);
            memcpy(idsMap, IdsData, sizeof(IdsData));
        }

        // SRV pour IDs
        BufferViewDesc idsViewDesc;
        idsViewDesc.ViewType = BUFFER_VIEW_SHADER_RESOURCE;
        idsViewDesc.Name = "Sphere IDs Buffer SRV";
        idsViewDesc.ByteOffset = 0;
        idsViewDesc.ByteWidth = sizeof(IdsData);
        m_pIDsBuffer->CreateView(idsViewDesc, &m_pSpheresIdsBufferView);
    }

    void SphereHandler::SetSphereBuffers(const rvtx::dil::SphereHolder2& holder)
    {
        auto* srb = pipelineEntry->SRB.RawPtr();
        if (!srb) { OutputDebugStringA("[SphereHandler] SRB is null.\n"); return; }

        auto* spheresSRV = holder.buffer.srv();
        auto* idsSRV = holder.idsBuffer.srv();
        if (!spheresSRV) OutputDebugStringA("[SphereHandler] spheres.srv() is null!\n");
        if (!idsSRV)     OutputDebugStringA("[SphereHandler] ids.srv() is null!\n");

        if (auto* v = srb->GetVariableByName(SHADER_TYPE_VERTEX, "spheres")) v->Set(spheresSRV);
        if (auto* v = srb->GetVariableByName(SHADER_TYPE_VERTEX, "ids"))     v->Set(idsSRV);

    }



    // ==========================================================
    // PIPELINE ENTRY
    // ==========================================================
    void SphereHandler::BindBuffers()
    {
        BufferDesc CBDesc;
        CBDesc.Name = "SphereSettings CB";
        CBDesc.Size = sizeof(SphereSettings);
        CBDesc.Usage = USAGE_DYNAMIC;
        CBDesc.BindFlags = BIND_UNIFORM_BUFFER;
        CBDesc.CPUAccessFlags = CPU_ACCESS_WRITE;  
        m_pDevice->CreateBuffer(CBDesc, nullptr, &m_pSphereSettingsCB);
        // Associer le constant buffer aux shaders
        auto* pVarVS = pipelineEntry->PSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SphereSettings");
        if (pVarVS != nullptr)
            pVarVS->Set(m_pSphereSettingsCB);

        auto* pVarGS = pipelineEntry->PSO->GetStaticVariableByName(SHADER_TYPE_GEOMETRY, "SphereSettings");
        if (pVarGS != nullptr)
            pVarGS->Set(m_pSphereSettingsCB);

        auto* pVarPS = pipelineEntry->PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SphereSettings");
        if (pVarPS != nullptr)
            pVarPS->Set(m_pSphereSettingsCB);





        pipelineEntry->PSO->CreateShaderResourceBinding(&pipelineEntry->SRB, true);

        
    }

    // ==========================================================
    // RENDU
    // ==========================================================
    void SphereHandler::Render(const float4x4& viewMatrix,
        const float4x4& projMatrix)
    {
        // Mettre à jour le constant buffer
        {
            MapHelper<SphereSettings> CBData(m_pImmediateContext, m_pSphereSettingsCB, MAP_WRITE, MAP_FLAG_DISCARD);
            CBData->uMVMatrix = viewMatrix;
            CBData->uProjMatrix = projMatrix;
            CBData->uRadiusAdd = 0.0f;
            CBData->uIsPerspective = 1;
        }

        // Activer le pipeline
        m_pImmediateContext->SetPipelineState(m_pPSO_Sphere);

        // Binder les buffers dynamiques
        auto varSpheres = m_pSRB_Sphere->GetVariableByName(SHADER_TYPE_VERTEX, "spheres");
        if (varSpheres) varSpheres->Set(m_pSpheresBufferView);

        auto varIDs = m_pSRB_Sphere->GetVariableByName(SHADER_TYPE_VERTEX, "ids");
        if (varIDs) varIDs->Set(m_pSpheresIdsBufferView);

        // Commit des ressources
        m_pImmediateContext->CommitShaderResources(m_pSRB_Sphere, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Draw
        DrawAttribs drawAttrs;
        drawAttrs.NumVertices = m_NumSpheres;
        drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
        m_pImmediateContext->Draw(drawAttrs);
    }

    void SphereHandler::RenderPE(const float4x4& viewMatrix,
        const float4x4& projMatrix)
    {
        // Mettre à jour le constant buffer
        {
            MapHelper<SphereSettings> CBData(m_pImmediateContext, m_pSphereSettingsCB, MAP_WRITE, MAP_FLAG_DISCARD);
            CBData->uMVMatrix = viewMatrix;
            CBData->uProjMatrix = projMatrix;
            CBData->uRadiusAdd = 0.0f;
            CBData->uIsPerspective = 1;
        }

        // Activer le pipeline
        m_pImmediateContext->SetPipelineState(pipelineEntry->PSO);
        //m_pImmediateContext->SetPipelineState(m_pPSO_Sphere);

        // Binder les buffers dynamiques
        auto varSpheres = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "spheres");
        if (varSpheres) varSpheres->Set(m_pSpheresBufferView);

        auto varIDs = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "ids");
        if (varIDs) varIDs->Set(m_pSpheresIdsBufferView);

        // Commit des ressources
        m_pImmediateContext->CommitShaderResources(pipelineEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Draw
        DrawAttribs drawAttrs;
        drawAttrs.NumVertices = m_NumSpheres;
        drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
        m_pImmediateContext->Draw(drawAttrs);
    }




    void SphereHandler::render(
        const Diligent::FirstPersonCamera& m_Camera,
        const Scene& scene)
    {
        const auto& viewMatrix = m_Camera.GetViewMatrix();
        const auto& projMatrix = m_Camera.GetProjMatrix();

        //Render(viewMatrix, projMatrix);
        RenderPE(viewMatrix, projMatrix);
    }

    glm::mat4 toGlm(const Diligent::float4x4& m)
    {
        glm::mat4 mat = glm::make_mat4(&m.m[0][0]);
        return glm::transpose(mat); // passe de row-major à column-major
    }

    inline Diligent::float4x4 ToDiligent_ColumnMajor(const glm::mat4& m)
    {
        Diligent::float4x4 out;
        std::memcpy(out.m, glm::value_ptr(m), 16 * sizeof(float));
        return out;
    }

    void SphereHandler::render2(const rvtx::Camera& cam, const Scene&)
    {
        const bool isGL = m_pDevice->GetDeviceInfo().IsGLDevice();

        glm::mat4 V = cam.getViewMatrix();
        glm::mat4 P = isGL
            ? glm::perspectiveRH_NO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar)  // GL
            : glm::perspectiveRH_ZO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar); // D3D/Vk

        auto Vd = ToDiligent_ColumnMajor(V);
        auto Pd = ToDiligent_ColumnMajor(P);
        RenderPE(Vd, Pd);
    }

    void SphereHandler::render_context(const rvtx::Camera& cam,
        const rvtx::Scene& scene,
        Diligent::IDeviceContext* ctx)
    {
        using namespace Diligent;




        // Matrices caméra
        const bool isGL = m_pDevice->GetDeviceInfo().IsGLDevice();
        const glm::mat4 V = cam.getViewMatrix();
        glm::mat4 P = isGL
            ? glm::perspectiveRH_NO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar)
            : glm::perspectiveRH_ZO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar);

        // <<< Flip Y uniquement hors OpenGL
        if (!isGL) {
            glm::mat4 flipY(1.0f);
            flipY[1][1] = -1.0f;      // multiplie la ligne/colonne Y par -1
            P = flipY * P;
        }

        ctx->SetPipelineState(pipelineEntry->PSO);

        // Pour chaque entité qui a Transform + SphereHolder2
        auto view = scene.registry.view<rvtx::Transform, rvtx::dil::SphereHolder2>();
        for (auto ent : view)
        {
            const auto& xf = view.get<rvtx::Transform>(ent);
            const auto& sh = view.get<rvtx::dil::SphereHolder2>(ent);
            if (sh.size == 0) continue;

            // --- Model matrix (T * R * S)
            const glm::mat4 T = glm::translate(glm::mat4(1.f), xf.position);
            const glm::mat4 R = glm::mat4_cast(xf.rotation);
            //const glm::mat4 S = glm::scale(glm::mat4(1.f), xf.scale);
            const glm::mat4 M = T * R;

            // --- Mettre à jour le constant buffer POUR CETTE ENTITÉ
            // IMPORTANT: DISCARd à chaque update (une zone mémoire fraîche par draw)
            {
                MapHelper<SphereSettings> cb(ctx, m_pSphereSettingsCB, MAP_WRITE, MAP_FLAG_DISCARD);
                cb->uMVMatrix = ToDiligent_ColumnMajor(V * M); // VS attend des positions en espace vue
                cb->uProjMatrix = ToDiligent_ColumnMajor(P);
                cb->uRadiusAdd = 0.0f;
                cb->uIsPerspective = 1;
            }

            // --- Binder les SRV pour VS et GS

            if (auto* v = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "spheres"))
                v->Set(sh.buffer.srv());
            if (auto* v = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "ids"))
                v->Set(sh.idsBuffer.srv());

            if (auto* g = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_GEOMETRY, "spheres"))
                g->Set(sh.buffer.srv());
            if (auto* g = pipelineEntry->SRB->GetVariableByName(SHADER_TYPE_GEOMETRY, "ids"))
                g->Set(sh.idsBuffer.srv());

            // --- Commit des ressources (indispensable surtout en Vulkan)
            ctx->CommitShaderResources(pipelineEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);


            // --- Un point par sphère, SV_VertexID indexe dans les StructuredBuffers
            DrawAttribs da{};
            if (useGeom) da.NumVertices = sh.size;
            else 
            {
                da.NumVertices = 4;
                da.NumInstances = sh.size;
            }
            da.Flags = DRAW_FLAG_VERIFY_ALL;
            ctx->Draw(da);
        }
    }



} // namespace rvtx::dil

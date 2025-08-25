

#pragma once

#include "rvtx/dil/geometry/ball_and_stick_handler.hpp"
#include "rvtx/dil/geometry/ball_and_stick_holder.hpp"
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


    BallAndStickHandler::BallAndStickHandler(PipelineManager& pipelineManager,
        RefCntAutoPtr<IRenderDevice> device,
        RefCntAutoPtr<IDeviceContext> context,
        RefCntAutoPtr<ISwapChain> swapChain,
        RefCntAutoPtr<IEngineFactory> engineFactory) :
        m_pDevice(device), m_pImmediateContext(context), m_pSwapChain(swapChain), m_pEngineFactory(engineFactory)
    {
        initializePSO();
        pipelineSphereEntry = pipelineManager.create2("SphereGeometry", { "shaders_hlsl/sphere/sphere.vsh",


            "shaders_hlsl/sphere/sphere.gsh" ,
            "shaders_hlsl/sphere/sphere.psh" }, PSOStateCreateInfo, VarsSphere, _countof(VarsSphere));

        //auto PSO_Cyl = PSOStateCreateInfo;
        //PSO_Cyl.PSODesc.Name = "Cylinder Impostor Pipeline";
        //PSO_Cyl.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_LINE_LIST;

        //pipelineCylinderEntry = pipelineManager.create2("CylinderGeometry", { "C:/M2 ISICG/Projet M2/final/rVTX/rvtx/gl/shaders_hlsl/cylinder/cylinder.vsh",
        //    "C:/M2 ISICG/Projet M2/final/rVTX/rvtx/gl/shaders_hlsl/cylinder/cylinder.gsh" ,
        //    "C:/M2 ISICG/Projet M2/final/rVTX/rvtx/gl/shaders_hlsl/cylinder/cylinder.psh" }, PSO_Cyl, VarsSphere, _countof(VarsSphere));
        BindBuffers();
    }

    // ==========================================================
    // PIPELINE MANAGER
    // ==========================================================


    void BallAndStickHandler::initializePSO()
    {
        PSOStateCreateInfo.PSODesc.Name = "Sphere Impostor Pipeline";
        PSOStateCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

        // --- Sorties du GBuffer ---
        PSOStateCreateInfo.GraphicsPipeline.NumRenderTargets = 3;
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[0] = TEX_FORMAT_RGBA32_UINT;   // positions+normales packées
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[1] = TEX_FORMAT_RGBA16_FLOAT;  // couleur/matériaux
        PSOStateCreateInfo.GraphicsPipeline.RTVFormats[2] = TEX_FORMAT_R32_UINT;      // IDs
        PSOStateCreateInfo.GraphicsPipeline.DSVFormat = TEX_FORMAT_D32_FLOAT;

        PSOStateCreateInfo.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_POINT_LIST;

        // Rasterizer / Depth
        PSOStateCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = True;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
        PSOStateCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthFunc = COMPARISON_FUNC_LESS;
    }

    void BallAndStickHandler::finalizePSO()
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
    // PIPELINE ENTRY
    // ==========================================================
    void BallAndStickHandler::BindBuffers()
    {
        BufferDesc CBDesc;
        CBDesc.Name = "SphereSettings CB";
        CBDesc.Size = sizeof(SphereSettings);
        CBDesc.Usage = USAGE_DYNAMIC;
        CBDesc.BindFlags = BIND_UNIFORM_BUFFER;
        CBDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
        m_pDevice->CreateBuffer(CBDesc, nullptr, &m_pSphereSettingsCB);
        // Associer le constant buffer aux shaders
        pipelineSphereEntry->PSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SphereSettings")->Set(m_pSphereSettingsCB);
        pipelineSphereEntry->PSO->GetStaticVariableByName(SHADER_TYPE_GEOMETRY, "SphereSettings")->Set(m_pSphereSettingsCB);
        pipelineSphereEntry->PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SphereSettings")->Set(m_pSphereSettingsCB);

        pipelineSphereEntry->PSO->CreateShaderResourceBinding(&pipelineSphereEntry->SRB, true);


    }

    // ==========================================================
    // RENDU
    // ==========================================================
    void BallAndStickHandler::Render(const float4x4& viewMatrix,
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
        drawAttrs.NumVertices = 0;
        drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
        m_pImmediateContext->Draw(drawAttrs);
    }

    void BallAndStickHandler::RenderPE(const float4x4& viewMatrix,
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
        m_pImmediateContext->SetPipelineState(pipelineSphereEntry->PSO);
        //m_pImmediateContext->SetPipelineState(m_pPSO_Sphere);

        // Binder les buffers dynamiques
        auto varSpheres = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "spheres");
        if (varSpheres) varSpheres->Set(m_pSpheresBufferView);

        auto varIDs = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "ids");
        if (varIDs) varIDs->Set(m_pSpheresIdsBufferView);

        // Commit des ressources
        m_pImmediateContext->CommitShaderResources(pipelineSphereEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Draw
        DrawAttribs drawAttrs;
        drawAttrs.NumVertices = 0;
        drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
        m_pImmediateContext->Draw(drawAttrs);
    }




    void BallAndStickHandler::render(
        const Diligent::FirstPersonCamera& m_Camera,
        const Scene& scene)
    {
        const auto& viewMatrix = m_Camera.GetViewMatrix();
        const auto& projMatrix = m_Camera.GetProjMatrix();

        //Render(viewMatrix, projMatrix);
        RenderPE(viewMatrix, projMatrix);
    }


    inline Diligent::float4x4 ToDiligent_ColumnMajor(const glm::mat4& m)
    {
        Diligent::float4x4 out;
        std::memcpy(out.m, glm::value_ptr(m), 16 * sizeof(float));
        return out;
    }

    void BallAndStickHandler::render2(const rvtx::Camera& cam, const Scene&)
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


    

    void BallAndStickHandler::render_context(const rvtx::Camera& cam,
        const rvtx::Scene& scene,
        Diligent::IDeviceContext* ctx)
    {
        using namespace Diligent;

        // Matrices caméra
        const bool isGL = m_pDevice->GetDeviceInfo().IsGLDevice();
        const glm::mat4 V = cam.getViewMatrix();
        const glm::mat4 P = isGL
            ? glm::perspectiveRH_NO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar)
            : glm::perspectiveRH_ZO(cam.fov, cam.getAspectRatio(), cam.zNear, cam.zFar);

        ctx->SetPipelineState(pipelineSphereEntry->PSO);

        // Pour chaque entité qui a Transform + SphereHolder2
        auto view = scene.registry.view<rvtx::Transform, rvtx::dil::BallAndStickHolder>();
        for (auto ent : view)
        {
            const auto& xf = view.get<rvtx::Transform>(ent);
            const auto& sh = view.get<rvtx::dil::BallAndStickHolder>(ent);
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
            if (auto* v = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "spheres"))
                v->Set(sh.buffer.srv());
            if (auto* v = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_VERTEX, "ids"))
                v->Set(sh.idsBuffer.srv());
            if (auto* g = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_GEOMETRY, "spheres"))
                g->Set(sh.buffer.srv());
            if (auto* g = pipelineSphereEntry->SRB->GetVariableByName(SHADER_TYPE_GEOMETRY, "ids"))
                g->Set(sh.idsBuffer.srv());

            ctx->CommitShaderResources(pipelineSphereEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            // --- Un point par sphère, SV_VertexID indexe dans les StructuredBuffers
            DrawAttribs da{};
            da.NumVertices = sh.size;
            da.Flags = DRAW_FLAG_VERIFY_ALL;
            ctx->Draw(da);
        }
    }



} // namespace rvtx::dil



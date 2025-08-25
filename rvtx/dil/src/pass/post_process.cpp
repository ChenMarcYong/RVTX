


#include "rvtx/dil/pass/post_process.hpp"

#include <Graphics/GraphicsEngine/interface/RenderDevice.h>
#include <Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <Graphics/GraphicsEngine/interface/Texture.h>
#include <Graphics/GraphicsEngine/interface/TextureView.h>
#include <Graphics/GraphicsEngine/interface/Shader.h>
#include <Graphics/GraphicsEngine/interface/PipelineState.h>
#include <Graphics/GraphicsEngine/interface/Buffer.h>
#include <Graphics/GraphicsEngine/interface/GraphicsTypes.h>
#include <Common/interface/RefCntAutoPtr.hpp>
#include <Graphics/GraphicsTools/interface/MapHelper.hpp>

#include "rvtx/dil/utils/pipeline_manager.hpp"
#include "rvtx/dil/geometry/types.hpp"

#include "rvtx/system/camera.hpp"

#include <Windows.h>
#include <filesystem>

#include <random>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <glm/gtc/type_ptr.hpp> // glm::value_ptr
#include "BasicMath.hpp"  

using namespace Diligent;

namespace rvtx::dil
{

     // 16B aligné
   

    struct DebugCBData { float Window[2]; float Gamma; Uint32 Invert; };
    struct DebugSizeCB { float2 InvRTSize; float2 pad; };

    // ==========================================================
    // LinearizeDepthPostProcessDiligent
    // ==========================================================

    LinearizeDepthPostProcessDiligent::LinearizeDepthPostProcessDiligent(uint32_t width, uint32_t height, PipelineManager& manager)
        : Pass{ width, height }, m_Manager(&manager)
    {
        Diligent::IRenderDevice* pDevice = getManager().m_pDevice;
        createTarget(pDevice, width, height);

        GraphicsPipelineStateCreateInfo PSOStateCreateInfo{};
        {
            
            auto& GP = PSOStateCreateInfo.GraphicsPipeline;

            // formats de la cible (R16F comme ton GL_R16F) et pas de depth
            GP.NumRenderTargets = 1;
            GP.RTVFormats[0] = static_cast<TEXTURE_FORMAT>(m_OutputFormat);   // <- ta RT linéarisée
            GP.DSVFormat = TEX_FORMAT_UNKNOWN;
            GP.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            GP.RasterizerDesc.CullMode = CULL_MODE_NONE;
            GP.DepthStencilDesc.DepthEnable = False;
            GP.BlendDesc.RenderTargets[0].BlendEnable = False;
        }

        ShaderResourceVariableDesc Vars[] = {
            {SHADER_TYPE_PIXEL, "gDepth",   SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}, // SRV (la depth)
            {SHADER_TYPE_PIXEL, "CameraCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC }  // cbuffer
        };



        m_PipelineEntry = m_Manager->create2("LinearizeDepth", { "shaders_hlsl/shading/linearize_depth.psh",
            "shaders_hlsl/full_screen.vsh"}, PSOStateCreateInfo, Vars, _countof(Vars));

        if (useDebug)
        {
            GraphicsPipelineStateCreateInfo PSOStateCreateInfoDebug{};
            {
                auto& GP = PSOStateCreateInfoDebug.GraphicsPipeline;
                GP.NumRenderTargets = 1;
                GP.RTVFormats[0] = m_Manager->m_pSwapChain->GetDesc().ColorBufferFormat; // backbuffer
                GP.DSVFormat = TEX_FORMAT_UNKNOWN;
                GP.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                GP.RasterizerDesc.CullMode = CULL_MODE_NONE;
                GP.DepthStencilDesc.DepthEnable = False;
                GP.BlendDesc.RenderTargets[0].BlendEnable = False;
            }


            ShaderResourceVariableDesc VarsDebug[] = {
                {SHADER_TYPE_PIXEL, "LinearDepthTex", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
                {SHADER_TYPE_PIXEL, "CameraCB",       SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
                {SHADER_TYPE_PIXEL, "DebugCB",        SHADER_RESOURCE_VARIABLE_TYPE_STATIC },
            };
            m_PipelineEntryDebug = m_Manager->create2("LinearizeDepthDebug", { "shaders_hlsl/shading/linearize_depth_debug.psh",
        "shaders_hlsl/full_screen_debug.vsh" }, PSOStateCreateInfoDebug, VarsDebug, _countof(VarsDebug));
        

            if (m_PipelineEntryDebug && m_PipelineEntryDebug->PSO)
            {


                RefCntAutoPtr<IBuffer> pDebugCB;
                {
                    //BufferDesc bd; bd.Name = "DebugCB"; bd.BindFlags = BIND_UNIFORM_BUFFER;
                    //bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = sizeof(DebugCBData);
                    
                    
                    BufferDesc bd;
                    bd.Name = "DebugCB";
                    bd.BindFlags = BIND_UNIFORM_BUFFER;
                    bd.Usage = USAGE_DYNAMIC;
                    bd.CPUAccessFlags = CPU_ACCESS_WRITE;
                    bd.Size = sizeof(DebugCBData);
                    
                    
                    getManager().m_pDevice->CreateBuffer(bd, nullptr, &m_pDebugCB);

                    // Ex: fenêtre sur les 20% proches + contraste fort
                    //MapHelper<DebugCBData> map(getManager().m_pImmediateContex, m_pDebugCB, MAP_WRITE, MAP_FLAG_DISCARD);
                    //map->Window[0] = 0.0f;   // NearDepth
                    //map->Window[1] = 0.03f;   // FarDepth   élargis la fenêtre
                    //map->Gamma = 0.6f;   // <1 = éclaircit, plus lisible
                    //map->Invert = 0;

                    m_PipelineEntryDebug->PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "DebugCB")->Set(m_pDebugCB);
                }

                m_PipelineEntryDebug->PSO->CreateShaderResourceBinding(&m_PipelineEntryDebug->SRB, true);

            }


        }

        

        if (m_PipelineEntry && m_PipelineEntry->PSO)
        {
            // 1) Créer le buffer constant
            BufferDesc cbDesc;
            cbDesc.Name = "CameraCB";
            cbDesc.BindFlags = BIND_UNIFORM_BUFFER;
            cbDesc.Usage = USAGE_DYNAMIC;
            cbDesc.CPUAccessFlags = CPU_ACCESS_WRITE;
            cbDesc.Size = sizeof(CameraCBData);
            pDevice->CreateBuffer(cbDesc, nullptr, &m_pCameraCB);

            // 2) Binder la variable STATIC sur le PSO (et pas le SRB)
            m_PipelineEntry->PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "CameraCB")
                ->Set(m_pCameraCB);

            // 3) Créer le SRB ensuite (contiendra gDepth qui est MUTABLE)
            m_PipelineEntry->PSO->CreateShaderResourceBinding(&m_PipelineEntry->SRB, true);

            // 4) Exemple de binding de la depth (si tu l’as sous forme de SRV)
            // m_PipelineEntry->SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gDepth")->Set(pDepthSRV);
        }


        //Camera cam;


    }

    void LinearizeDepthPostProcessDiligent::createTarget(Diligent::IRenderDevice* pDevice,
        uint32_t w,
        uint32_t h)
    {
        using namespace Diligent;

        // Format de sortie (par défaut R32F dans le .hpp, change si besoin)

        TextureDesc desc{};
        desc.Name = "LinearDepth.Target";
        desc.Type = RESOURCE_DIM_TEX_2D;
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = 1;
        desc.SampleCount = 1;
        desc.Format = static_cast<TEXTURE_FORMAT>(m_OutputFormat);
        desc.Usage = USAGE_DEFAULT;
        desc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;

        // Libère l’ancienne (si existait) puis recrée
        m_Output.Release();
        m_OutputRTV.Release();
        m_OutputSRV.Release();

        Diligent::RefCntAutoPtr<ITexture> tex;
        pDevice->CreateTexture(desc, nullptr, &tex);
        m_Output = std::move(tex);
        m_OutputRTV = m_Output->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_OutputSRV = m_Output->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    void LinearizeDepthPostProcessDiligent::Execute(
        Diligent::IDeviceContext* ctx,
        const rvtx::Camera& camera,
        Diligent::ITextureView* depthSRV)
    {



        const auto w = m_Output->GetDesc().Width;
        const auto h = m_Output->GetDesc().Height;

        Viewport vp1{ 0,0, float(w), float(h), 0, 1 };
        ctx->SetViewports(1, &vp1, w, h);

        ITextureView* RTVs[] = { m_OutputRTV };
        ctx->SetRenderTargets(1, RTVs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // MAJ CameraCB
        MapHelper<CameraCBData> map(ctx, m_pCameraCB, MAP_WRITE, MAP_FLAG_DISCARD);
        map->uClipInfo[0] = camera.zNear * camera.zFar;
        map->uClipInfo[1] = camera.zNear - camera.zFar;
        map->uClipInfo[2] = camera.zFar;
        map->uClipInfo[3] = camera.zNear;
        map->uIsPerspective = (camera.projectionType == rvtx::Camera::Projection::Perspective ? 1u : 0u);

        // MAJ SRV de profondeur
        m_PipelineEntry->SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gDepth")->Set(depthSRV);

        // Set pipeline et RT
        ctx->SetPipelineState(m_PipelineEntry->PSO);
        ctx->CommitShaderResources(m_PipelineEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        //ITextureView* RTVs[] = { m_OutputRTV };
        //ctx->SetRenderTargets(1, RTVs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        //const float green[4] = { 0,1,0,1 };
        //ctx->SetRenderTargets(1, &m_OutputRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        //ctx->ClearRenderTarget(m_OutputRTV, green, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);



        // Dessin fullscreen
        DrawAttribs attrs{};
        attrs.NumVertices = 3;
        attrs.Flags = DRAW_FLAG_VERIFY_ALL;
        ctx->Draw(attrs);

        if(useDebug && m_pDebugCB)
        {
            m_PipelineEntryDebug->SRB->GetVariableByName(SHADER_TYPE_PIXEL, "LinearDepthTex")->Set(m_OutputSRV);

            // Cible = backbuffer
            auto* pRTV = getManager().m_pSwapChain->GetCurrentBackBufferRTV();
            getManager().m_pSwapChain->GetCurrentBackBufferRTV();

            ITextureView* RTs[] = { pRTV };
            ctx->SetRenderTargets(1, RTs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);


            m_PipelineEntryDebug->SRB
                ->GetVariableByName(SHADER_TYPE_PIXEL, "LinearDepthTex")
                ->Set(m_OutputSRV);

            MapHelper<DebugCBData> map(ctx, m_pDebugCB, MAP_WRITE, MAP_FLAG_DISCARD);
            *map = m_DebugCBData;


            // PSO + ressources
            ctx->SetPipelineState(m_PipelineEntryDebug->PSO);
            ctx->CommitShaderResources(m_PipelineEntryDebug->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

            // Full-screen triangle
            DrawAttribs DA{ 3, DRAW_FLAG_VERIFY_ALL };
            ctx->Draw(DA);
        }



        //const float pink[4] = { 1,0,1,1 };
        //ctx->SetRenderTargets(1, &pRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        //ctx->ClearRenderTarget(pRTV, pink, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);


        OutputDebugStringA("[LinearDepth] Debug pass draw\n");

    }



    void LinearizeDepthPostProcessDiligent::render(
    const rvtx::Camera& camera)
{

     Diligent::IDeviceContext* ctx = getManager().m_pImmediateContex;

    // --- MAJ du CB avec les paramètres actuels de la caméra ---
    MapHelper<CameraCBData> map(ctx, m_pCameraCB, MAP_WRITE, MAP_FLAG_DISCARD);
    map->uClipInfo[0] = camera.zNear * camera.zFar;
    map->uClipInfo[1] = camera.zFar;
    map->uClipInfo[2] = camera.zFar - camera.zNear;
    map->uClipInfo[3] = camera.zNear;
    map->uIsPerspective = (camera.projectionType == rvtx::Camera::Projection::Perspective ? 1u : 0u);

    // --- MAJ du SRV de profondeur ---
    m_PipelineEntry->SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gDepth")->Set(m_OutputSRV);

    // --- Dessin plein écran ---
    ITextureView* pRTVs[] = { m_OutputRTV };
    ctx->SetRenderTargets(1, pRTVs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    ctx->SetPipelineState(m_PipelineEntry->PSO);
    ctx->CommitShaderResources(m_PipelineEntry->SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    DrawAttribs drawAttrs;
    drawAttrs.NumVertices = 3;
    drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
    ctx->Draw(drawAttrs);
}


    void LinearizeDepthPostProcessDiligent::resize(IRenderDevice* device, uint32_t w, uint32_t h)
    {

        // Ici : recréer les ressources communes si besoin
    }

    // ==========================================================
    // SSAOPostProcessDiligent
    // ==========================================================


    struct SSAOParamsCB
    {
        Diligent::float4x4 uProjMatrix; // row-major côté C++ si ton HLSL est row_major
        Diligent::float4x4 uInvProjMatrix; // row-major côté C++ si ton HLSL est row_major
        int   uAoIntensity = 2;
        int   uKernelSize = 16;
        float uNoiseSize = 64.f;
        float _pad0 = 0.f;
        float2 RTSize, InvRTSize;
    };






    SSAOPostProcessDiligent::SSAOPostProcessDiligent(uint32_t width, uint32_t height, PipelineManager& manager)
        : Pass{ width, height }
        , m_Manager{ &manager }
    {
        Diligent::IRenderDevice* pDevice = getManager().m_pDevice;
        
        createTarget(manager.m_pDevice, width, height);

        generateKernel();

        createNoiseTexture(manager.m_pDevice, m_NoiseTextureSize);

        OutputDebugStringA("[POSTPROCESS] Load postProcess.\n");

        auto scDesc = m_Manager->m_pSwapChain->GetDesc();

        GraphicsPipelineStateCreateInfo PsoCI{};
        auto& GP = PsoCI.GraphicsPipeline;
        GP.NumRenderTargets = 1;
        GP.RTVFormats[0] = m_Manager->m_pSwapChain->GetDesc().ColorBufferFormat; //TEX_FORMAT_R8_UNORM;     // même que m_Output
        GP.DSVFormat = TEX_FORMAT_UNKNOWN;
        GP.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        GP.RasterizerDesc.CullMode = CULL_MODE_NONE;
        GP.DepthStencilDesc.DepthEnable = False;
        GP.BlendDesc.RenderTargets[0].BlendEnable = False;

        // Ressources (noms = ceux attendus par ton shader SSAO HLSL/GLSL)
        ShaderResourceVariableDesc Vars[] = {
            {SHADER_TYPE_PIXEL, "gViewPosNormal", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_PIXEL, "gNoise",         SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
            {SHADER_TYPE_PIXEL, "gLinearDepth",   SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        };

        PsoCI.PSODesc.ResourceLayout.NumVariables = static_cast<Uint32>(std::size(Vars));
        PsoCI.PSODesc.ResourceLayout.Variables = Vars;
        // Samplers statiques (équiv. GL_NEAREST/CLAMP pour depth/geom, REPEAT pour noise)

        SamplerDesc sampPointClamp{};
        sampPointClamp.MinFilter = sampPointClamp.MagFilter = sampPointClamp.MipFilter = FILTER_TYPE_POINT;
        sampPointClamp.AddressU = sampPointClamp.AddressV = TEXTURE_ADDRESS_CLAMP;


        SamplerDesc sampPointWrap = sampPointClamp;
        sampPointWrap.AddressU = sampPointWrap.AddressV = TEXTURE_ADDRESS_WRAP;

        const ImmutableSamplerDesc Imtbl[] = {
            {SHADER_TYPE_PIXEL, "gNoiseSampler",  sampPointWrap},
            {SHADER_TYPE_PIXEL, "gDepthSampler",  sampPointClamp},
        };

        PsoCI.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)std::size(Imtbl);
        PsoCI.PSODesc.ResourceLayout.ImmutableSamplers = Imtbl;


        auto* entry = m_Manager->create2(
            "SSAO",
            {
                "shaders_hlsl/shading/ssao_debug2.psh",
                "shaders_hlsl/full_screen.vsh"
            },
            PsoCI, Vars,
            _countof(Vars));

        m_PSO = entry->PSO;
        createCBuffers(manager.m_pDevice);
        m_PSO->CreateShaderResourceBinding(&m_SRB, true);

        
    };

    void SSAOPostProcessDiligent::createTarget(Diligent::IRenderDevice* pDevice,
        uint32_t w,
        uint32_t h)
    {
        using namespace Diligent;

        m_width = w;
        m_height = h;

        // Libère l'ancienne cible si elle existe
        m_Output.Release();
        m_OutputRTV.Release();
        m_OutputSRV.Release();

        // Décrit la texture AO (éq. glTexImage2D(GL_R8, ...))
        TextureDesc desc{};
        desc.Name = "SSAO.Output";
        desc.Type = RESOURCE_DIM_TEX_2D;
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = 1;
        desc.SampleCount = 1;
        desc.Format = m_OutputFormat; // ex: TEX_FORMAT_R8_UNORM (par défaut dans le .hpp)
        desc.Usage = USAGE_DEFAULT;
        desc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;

        // Crée la texture et récupère ses vues
        pDevice->CreateTexture(desc, nullptr, &m_Output);
        m_OutputRTV = m_Output->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_OutputSRV = m_Output->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    void SSAOPostProcessDiligent::createCBuffers(Diligent::IRenderDevice* pDevice)
    {
        using namespace Diligent;

        // b0 : SSAOParams
        BufferDesc cbd{};
        cbd.Name = "SSAO.ParamsCB";
        cbd.Size = sizeof(SSAOParamsCB);        // struct ci-dessous
        cbd.Usage = USAGE_DYNAMIC;
        cbd.BindFlags = BIND_UNIFORM_BUFFER;
        cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
        pDevice->CreateBuffer(cbd, nullptr, &m_CB);

        // b1 : SSAOKernel (max 512 float3)
        BufferDesc kbd{};
        kbd.Name = "SSAO.KernelCB";
        kbd.Size = sizeof(float) * 3 * 512;
        kbd.Usage = USAGE_DYNAMIC;
        kbd.BindFlags = BIND_UNIFORM_BUFFER;
        kbd.CPUAccessFlags = CPU_ACCESS_WRITE;
        pDevice->CreateBuffer(kbd, nullptr, &m_CBKernel);

        BufferDesc db{};    //debug
        db.Name = "SSAO.DebugSize";
        db.Size = sizeof(DebugSizeCB);
        db.Usage = USAGE_DYNAMIC;
        db.BindFlags = BIND_UNIFORM_BUFFER;
        db.CPUAccessFlags = CPU_ACCESS_WRITE;
        pDevice->CreateBuffer(db, nullptr, &m_CBDebug);



        // Binder les CBs statiquement sur le PSO (PIXEL)
        m_PSO->GetStaticVariableByName(Diligent::SHADER_TYPE_PIXEL, "SSAOParams")->Set(m_CB);
        if (auto* var = m_PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SSAOKernel"))
        {
            std::string msg = "[KERNEL] works : \n";

            OutputDebugStringA(msg.c_str());
            var->Set(m_CBKernel);
        }
            
        //m_PSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "DebugSize")->Set(m_CBDebug);
    }


    inline Diligent::float4x4 ToFloat4x4RowMajor(const glm::mat4& g)
    {
        Diligent::float4x4 m;               // m.m[4][4] est dispo dans BasicMath.hpp
        const float* a = glm::value_ptr(g); // a[c*4 + r]
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m.m[r][c] = a[c * 4 + r];   // transpose-copy
        return m;
    }

    // Variante si ton cbuffer n’est PAS row_major (peu probable ici) :
    inline Diligent::float4x4 ToFloat4x4ColumnMajor(const glm::mat4& g)
    {
        Diligent::float4x4 m;
        const float* a = glm::value_ptr(g);
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m.m[r][c] = a[r * 4 + c];   // copie directe
        return m;
    }


    void SSAOPostProcessDiligent::render(const rvtx::Camera& camera)
    {
        Diligent::IDeviceContext* ctx = getManager().m_pImmediateContex;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain> swap = getManager().m_pSwapChain;

        // MAJ CBs (OK)
        {
            MapHelper<SSAOParamsCB> map(ctx, m_CB, MAP_WRITE, MAP_FLAG_DISCARD);
            map->uProjMatrix = ToFloat4x4ColumnMajor(camera.getProjectionMatrix());
            map->uInvProjMatrix = ToFloat4x4ColumnMajor(glm::inverse(camera.getProjectionMatrix()));
            map->uAoIntensity = m_AOIntensity;
            map->uKernelSize = (int)m_KernelSize;
            map->uNoiseSize = float(m_NoiseTextureSize);

            map->RTSize = float2(float(m_width), float(m_height));
            map->InvRTSize = float2(1.0f / m_width, 1.0f / m_height);

        }
        {
            MapHelper<float> map(ctx, m_CBKernel, MAP_WRITE, MAP_FLAG_DISCARD);
            std::memcpy(map, m_AOKernel.data(), sizeof(float) * 3 * m_KernelSize);
        }

        // Viewport pour la RT SSAO
        Diligent::Viewport vp{ 0,0,float(m_width),float(m_height),0,1 };
        ctx->SetViewports(1, &vp, m_width, m_height);





        // RT SSAO + PSO AO
        ITextureView* rtvs[] = { m_OutputRTV };
        ctx->SetRenderTargets(1, rtvs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        ctx->SetPipelineState(m_PSO);

        // Bind SRV correctes
        if (m_SRB)
        {
            if (m_ViewPosNormalSRV)
                m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gViewPosNormal")->Set(m_ViewPosNormalSRV);
            if (m_NoiseSRV)
                m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gNoise")->Set(m_NoiseSRV);
            if (m_LinearDepthSRV)
                m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gLinearDepth")->Set(m_LinearDepthSRV);


            ctx->CommitShaderResources(m_SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        DrawAttribs da{ 3, DRAW_FLAG_VERIFY_ALL };
        ctx->Draw(da);
    }



    void SSAOPostProcessDiligent::renderToBackBuffer(const rvtx::Camera& camera)
    {



        Diligent::IDeviceContext* ctx = getManager().m_pImmediateContex;


        auto* pRTV = getManager().m_pSwapChain->GetCurrentBackBufferRTV();
        ctx->SetRenderTargets(1, &pRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // >> viewport = taille du backbuffer (et pas celle de m_Output)
        const auto bbDesc = pRTV->GetTexture()->GetDesc();
        Viewport vp{ 0, 0, float(bbDesc.Width), float(bbDesc.Height), 0, 1 };
        ctx->SetViewports(1, &vp, bbDesc.Width, bbDesc.Height);
        IRenderDevice*  m_Device = getManager().m_pDevice;
        resize(m_Device, bbDesc.Width, bbDesc.Height);

         //MAJ des CB
        {
            MapHelper<SSAOParamsCB> map(ctx, m_CB, MAP_WRITE, MAP_FLAG_DISCARD);
            map->uProjMatrix = ToFloat4x4ColumnMajor(camera.getProjectionMatrix());
            map->uInvProjMatrix = ToFloat4x4ColumnMajor(glm::inverse(camera.getProjectionMatrix()));
            map->uAoIntensity = m_AOIntensity;
            map->uKernelSize = static_cast<int>(m_KernelSize);
            map->uNoiseSize = static_cast<float>(m_NoiseTextureSize);
            map->RTSize = float2(float(m_width), float(m_height));
            map->InvRTSize = float2(1.0f / m_width, 1.0f / m_height);
        }

        std::string msg = "[SSAO] m_width : " + std::to_string(float2(float(m_width), float(m_height)).x)
            + " m_height : " + std::to_string(float2(float(m_width), float(m_height)).y) + "\n";

        OutputDebugStringA(msg.c_str());


        std::vector<float> tmp(4 * m_KernelSize);
        for (uint32_t i = 0; i < m_KernelSize; ++i) {
            tmp[4 * i + 0] = m_AOKernel[i].x;
            tmp[4 * i + 1] = m_AOKernel[i].y;
            tmp[4 * i + 2] = m_AOKernel[i].z;
            tmp[4 * i + 3] = 0.0f;
        }


        {
            MapHelper<float> map(ctx, m_CBKernel, MAP_WRITE, MAP_FLAG_DISCARD);
            std::memcpy(map, tmp.data(), sizeof(float) * 4 * m_KernelSize);
        }

        std::string msg2 = "[SSAO] m_AOKernel size : " + std::to_string(m_AOKernel.size())+"\n";
        OutputDebugStringA(msg2.c_str());
        //{ // debug
        //    MapHelper<DebugSizeCB> map(ctx, m_CBDebug, MAP_WRITE, MAP_FLAG_DISCARD);
        //    map->InvRTSize = float2(1.0f / m_width, 1.0f / m_height);
        //}


        // PSO debug + SRV correctes
        ctx->SetPipelineState(m_PSO); // <-- ton PSO unique pour debug
        if (m_SRB)
        {
            if (m_ViewPosNormalSRV)
            {
                m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gViewPosNormal")->Set(m_ViewPosNormalSRV);
            }
                
            if (m_LinearDepthSRV) // SRV de la passe LinearizeDepth !
                m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gLinearDepth")->Set(m_LinearDepthSRV);
            if (m_NoiseSRV)
                if (auto* v = m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "gNoise"))
                    v->Set(m_NoiseSRV);

            ctx->CommitShaderResources(m_SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        auto* tex = m_ViewPosNormalSRV ? m_ViewPosNormalSRV->GetTexture() : nullptr;
            if (tex)
            {
                const auto& d = tex->GetDesc();
                std::string s = "[VP/N] fmt=" + std::to_string((int)d.Format) +
                    " size=" + std::to_string(d.Width) + "x" + std::to_string(d.Height) + "\n";
                OutputDebugStringA(s.c_str());
            }


        // triangle plein écran
        Diligent::DrawAttribs da{ 3, Diligent::DRAW_FLAG_VERIFY_ALL };
        ctx->Draw(da);
    }


    // ---------------- SSAOPostProcessDiligent ----------------

    void SSAOPostProcessDiligent::resize(IRenderDevice* pDevice,
        uint32_t w, uint32_t h)
    {
        if (!pDevice || w == 0 || h == 0)
            return;

        // Taille déjà à jour ?
        if (m_Output && m_Output->GetDesc().Width == w &&
            m_Output->GetDesc().Height == h)
            return;

        // Mémorise la nouvelle taille et recrée la cible AO
        createTarget(pDevice, w, h);

        // Pas besoin de recréer le SRB ni les PSO :
        //  - les CBs sont statiques sur le PSO,
        //  - le SRB ne référence pas la RT AO (uniquement des SRV d’entrée),
        //  - RTSize/InvRTSize sont remis à jour dans render() à chaque frame.
    }



    void SSAOPostProcessDiligent::generateKernel()
    {
        m_AOKernel.clear();
        m_AOKernel.resize(m_KernelSize);

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::mt19937 rng{ std::random_device{}() };

        for (uint32_t i = 0; i < m_KernelSize; i++)
        {
            // échantillon sur l’hémisphère orientée Z+
            glm::vec3 sample = glm::sphericalRand(1.0f); // vecteur aléatoire sur la sphère
            sample.z = std::abs(sample.z);               // garder côté hémisphère positif

            // appliquer un facteur aléatoire pour l'éloignement radial
            sample *= dist(rng);

            // interpolation progressive (points plus denses près du centre)
            float scale = static_cast<float>(i) / static_cast<float>(m_KernelSize);
            scale = glm::mix(0.01f, 1.0f, scale * scale);

            sample *= scale;
            m_AOKernel[i] = sample;
        }
    }


    void SSAOPostProcessDiligent::createNoiseTexture(Diligent::IRenderDevice* pDevice,
        uint32_t size)
    {
        using namespace Diligent;

        // Sécurité
        if (size == 0) size = 4;
        m_Noise.Release();
        m_NoiseSRV.Release();
        m_NoiseTextureSize = size;

        // 1) Générer les vecteurs 2D aléatoires normalisés (z = 0)
        //    (comme ta version GL : vec3(rand(-1,1), rand(-1,1), 0) normalisé)
        std::mt19937 rng{ std::random_device{}() };
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> noise; // RG16F => 2 floats par texel
        noise.resize(static_cast<size_t>(size) * size * 2);

        for (uint32_t y = 0; y < size; ++y)
        {
            for (uint32_t x = 0; x < size; ++x)
            {
                float vx = dist(rng);
                float vy = dist(rng);
                float len = std::sqrt(vx * vx + vy * vy);
                if (len < 1e-6f) { vx = 1.0f; vy = 0.0f; }
                else { vx /= len; vy /= len; }

                size_t idx = (static_cast<size_t>(y) * size + x) * 2;
                noise[idx + 0] = vx;
                noise[idx + 1] = vy;
            }
        }

        // 2) Décrire la texture (RG16F, SRV)
        TextureDesc desc{};
        desc.Name = "SSAO.Noise";
        desc.Type = RESOURCE_DIM_TEX_2D;
        desc.Width = size;
        desc.Height = size;
        desc.MipLevels = 1;
        desc.SampleCount = 1;
        desc.Format = TEX_FORMAT_RG16_FLOAT; // compact et suffisant pour (x,y)
        desc.Usage = USAGE_IMMUTABLE;       // le bruit ne change pas après création
        desc.BindFlags = BIND_SHADER_RESOURCE;

        // 3) Données initiales
        TextureSubResData sub{};
        sub.pData = noise.data();
        sub.Stride = size * sizeof(float) * 2; // row pitch (bytes)

        TextureData init{};
        init.pSubResources = &sub;
        init.NumSubresources = 1;

        // 4) Créer la texture + SRV
        RefCntAutoPtr<ITexture> tex;
        pDevice->CreateTexture(desc, &init, &tex);
        m_Noise = std::move(tex);
        m_NoiseSRV = m_Noise->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

    }


    // ==========================================================
    // PostProcessPassDiligent
    // ==========================================================




    PostProcessPassDiligent::PostProcessPassDiligent(uint32_t width,
        uint32_t height,
        PipelineManager& manager)
        : Pass(width, height),
        m_linearizeDepth(width, height, manager), // appelle le ctor de Pass
        m_ssao(width, height, manager),
        m_PipelineManager(&manager)
    {

    }

    void PostProcessPassDiligent::resize(IRenderDevice* device, uint32_t w, uint32_t h)
    {

        // Ici : recréer les ressources communes si besoin
    }

}



#include "rvtx/dil/renderer.hpp"

#include <Graphics/GraphicsEngine/interface/RenderDevice.h>
#include <Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <Graphics/GraphicsEngine/interface/SwapChain.h>
#include <Graphics/GraphicsTools/interface/MapHelper.hpp>

#include "rvtx/core/logger.hpp"

#include <Windows.h>
#include <filesystem>

#include "rvtx/dil/pass/post_process.hpp"

using namespace Diligent;

static const char* kLightingVS = R"(
struct VSOut { float4 Pos : SV_POSITION; float2 UV : TEXCOORD0; };
VSOut main(uint vid : SV_VertexID)
{
    float2 pos[3] = { float2(-1,-1), float2(-1,3), float2(3,-1) };
    float2 uv[3]  = { float2(0,0),   float2(0,2),   float2(2,0)  };
    VSOut o;
    o.Pos = float4(pos[vid], 0, 1);
    o.UV  = uv[vid];
    return o;
}
)";

// PS “composition” minimal : affiche la cible couleur du GBuffer.
// Les autres textures sont déjà bindées pour évoluer vers un vrai lighting.
static const char* kLightingPSLambert = R"(
Texture2D<uint4>  g_PosNorm : register(t0);
Texture2D<float4> g_Color   : register(t1);
SamplerState      g_ColorSampler;
Texture2D<uint>   g_ID      : register(t2);

struct VSOut { float4 Pos : SV_POSITION; float2 UV : TEXCOORD0; };

// helpers pour décoder l'octa normal encodé en 2x uint16
static float2 SignNotZero(float2 v){ return float2(v.x >= 0 ? 1.0 : -1.0, v.y >= 0 ? 1.0 : -1.0); }

float3 DecodeNormalOct16(uint2 enc16)
{
    // [0..65535] -> [-1..1]
    float2 f = (float2(enc16) / 65535.0) * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0)
    {
        float2 s = SignNotZero(n.xy);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

float4 main(VSOut i) : SV_TARGET
{
    // coordonnées pixel (pour les textures UINT -> Load)
    uint w, h;
    g_Color.GetDimensions(w, h); // on prend ses dimensions
    uint2 pix = uint2(saturate(i.UV) * float2(w, h));
    pix = min(pix, uint2(w - 1, h - 1));

    uint4 pn = g_PosNorm.Load(int3(pix, 0));
    float4 col = g_Color  .Load(int3(pix,0));

    if (pn.x == 0u)
        return col; // fond = blanc (car g_Color a été clear à blanc)


    uint  id = g_ID.Load(int3(pix, 0));

    // si rien n'a été écrit dans le GBuffer (clear=0), on retourne noir
    if (pn.x == 0u)
        return float4(0,0,0,1);

    float3 N = DecodeNormalOct16(pn.yz);

    // albédo via sampler combiné (Diligent: g_ColorSampler)
    float4 albedo = g_Color.Sample(g_ColorSampler, i.UV);

    // lumière directionnelle simple (espace vue)
    float3 L = normalize(float3(0.4, 0.6, -0.7));
    float  NdotL = saturate(dot(N, L));
    float  ambient = 0.1;

    float3 lit = albedo.rgb * (ambient + NdotL);
    return float4(lit, 1.0);

}
)";


namespace rvtx::dil
{
    DiligentRenderer3::DiligentRenderer3(IRenderDevice* dev,
        IDeviceContext* ctx,
        ISwapChain* swap,
        rvtx::dil::PipelineManager& pipeline,
        uint32_t width,
        uint32_t height)
        : m_Device{ dev }, m_Ctx{ ctx }, m_Swap{ swap }, m_Pipeline{ &pipeline}, m_Width{width}, m_Height{height}
    {
        // Création du GBuffer
        m_Pipeline->m_pDevice = m_Device;
        m_Pipeline->m_pImmediateContex = m_Ctx;
        m_Pipeline->m_pSwapChain = m_Swap;


        m_GBuffer = std::make_unique<GBufferPass>(m_Device, m_Width, m_Height);
        m_postProcessPass = std::make_unique<PostProcessPassDiligent>( m_Width, m_Height, *m_Pipeline);
        
        //auto post = std::make_unique<rvtx::dil::PostProcessPassDiligent>(
        //    m_Width, m_Height, *m_Pipeline   // <—
        //);
        
        CreateLightingPipeline();
        // Création de la RT finale (où on va blitter/postprocess)
        CreateTargets();

        // PSO pour le postprocess (simple copy fullscreen quad)
        CreatePostPresentPipeline();
    }

    void DiligentRenderer3::Resize(uint32_t width, uint32_t height)
    {
        if (width == m_Width && height == m_Height)
            return;

        m_Width = width;
        m_Height = height;

        // Resize du GBuffer
        if (m_GBuffer)
            m_GBuffer->Resize(m_Device, width, height);

        // Resize de la RT finale
        CreateTargets();
    }


    void DiligentRenderer3::Render(const rvtx::Camera& cam,
        const rvtx::Scene& scene,
        const std::function<void()>& updateUI)
    {
        // 1) GBuffer : on remplit les attachments (dont la couleur/albédo)



        m_GBuffer->Render(m_Ctx, [&] {
            if (m_Geometry)
                m_Geometry->render_context(cam, scene, m_Ctx);
            });

        if (m_postProcessPass) {
            m_postProcessPass->m_linearizeDepth.Execute(m_Ctx, cam, m_GBuffer->GetDepthSRV());

            m_postProcessPass->m_ssao.setViewPosNormalSRV(m_GBuffer->GetGeometrySRV());
            m_postProcessPass->m_ssao.setLinearDepthSRV(m_postProcessPass->m_linearizeDepth.getSRV());

            m_postProcessPass->m_ssao.renderToBackBuffer(cam);
            return; // << ne fais rien après, sinon tu recouvres
        }



        //if (m_postProcessPass)
        //{
        //    m_postProcessPass->m_linearizeDepth.Execute(m_Ctx, cam, m_GBuffer->GetDepthSRV());
        //}


        // 2) Composition minimale : afficher la RT couleur du GBuffer
        if (!m_LightingPSO) {
            OutputDebugStringA("[Render] Lighting PSO is null (creation failed). Aborting draw.\n");
            return; // évite l'assert SetPipelineState(nullptr)
        }
        if (!m_LightingSRB) {
            m_LightingPSO->CreateShaderResourceBinding(&m_LightingSRB, true);
            if (!m_LightingSRB) {
                OutputDebugStringA("[Render] Failed to create Lighting SRB.\n");
                return;
            }
        }

        const auto& sc = m_Swap->GetDesc();

        Viewport vp{};
        vp.TopLeftX = 0.0f;
        vp.TopLeftY = 0.0f;
        vp.Width = static_cast<float>(sc.Width);
        vp.Height = static_cast<float>(sc.Height);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        m_Ctx->SetViewports(1, &vp, sc.Width, sc.Height);


        Rect scissor{ 0, 0, static_cast<Int32>(sc.Width), static_cast<Int32>(sc.Height) };
        m_Ctx->SetScissorRects(1, &scissor, sc.Width, sc.Height);

        ITextureView* bbRTV = m_Swap->GetCurrentBackBufferRTV();
        m_Ctx->SetRenderTargets(1, &bbRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        m_Ctx->SetPipelineState(m_LightingPSO);

        // On ne lit que la couleur (t1) dans le PS minimal

        auto bind = [&](const char* name, ITextureView* srv) {
            if (auto* v = m_LightingSRB->GetVariableByName(SHADER_TYPE_PIXEL, name))
                v->Set(srv);
            else
                OutputDebugStringA((std::string("[Lighting] PS var not found: ") + name + "\n").c_str());
            };


        bind("g_PosNorm", m_GBuffer->GetGeometrySRV());
        bind("g_Color", m_GBuffer->GetMaterialSRV());
        bind("g_ID", m_GBuffer->GetIdSRV());


        m_Ctx->CommitShaderResources(m_LightingSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);




        // Fullscreen triangle via SV_VertexID
        DrawAttribs da{};
        da.NumVertices = 3;
        da.Flags = DRAW_FLAG_VERIFY_ALL;
        m_Ctx->Draw(da);

        // 3) UI éventuelle...

        // 4) Présentation
        //m_Swap->Present();
    }



    void DiligentRenderer3::Render2(const rvtx::Camera& cam,
        const rvtx::Scene& scene,
        const std::function<void()>& updateUI)
    {
        // Variante : bypass postprocess, dessiner directement la géométrie sur le swapchain

        auto* backRTV = m_Swap->GetCurrentBackBufferRTV();
        auto* backDSV = m_Swap->GetDepthBufferDSV();

        const float clearCol[] = { m_ClearColor.x, m_ClearColor.y, m_ClearColor.z, m_ClearColor.w };
        m_Ctx->ClearRenderTarget(backRTV, clearCol, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        m_Ctx->ClearDepthStencil(backDSV, CLEAR_DEPTH_FLAG, 1.0f, 0,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        m_Ctx->SetRenderTargets(1, &backRTV, backDSV,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        if (m_Geometry)
        {
            m_Geometry->render_context(cam, scene, m_Ctx);
        }

        if(m_postProcessPass)
        {
            m_postProcessPass->m_linearizeDepth.Execute(m_Ctx, cam, m_GBuffer->GetDepthSRV());
        }

        m_Swap->Present();
    }

    // ===================== PRIVES =====================

    void DiligentRenderer3::CreateTargets()
    {
        // Render target finale (RGBA16_FLOAT par ex.)
        TextureDesc texDesc;
        texDesc.Name = "FinalRT";
        texDesc.Type = RESOURCE_DIM_TEX_2D;
        texDesc.Width = m_Width;
        texDesc.Height = m_Height;
        texDesc.MipLevels = 1;
        texDesc.Format = TEX_FORMAT_RGBA16_FLOAT;
        texDesc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;

        m_Device->CreateTexture(texDesc, nullptr, &m_FinalTex);
        m_FinalRTV = m_FinalTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_FinalSRV = m_FinalTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    void DiligentRenderer3::CreatePostPresentPipeline()
    {
        // PSO minimal fullscreen quad pour copier depuis SRV -> RTV
        // À compléter avec tes shaders (fullscreen VS + copy PS)
        // Ici je laisse la création vide si tu n’as pas encore les shaders.
    }


    void DiligentRenderer3::CreateLightingPipeline()
    {
        using namespace Diligent;

        RefCntAutoPtr<IShader> vs, ps;

        // --- Shaders ---
        ShaderCreateInfo sci{};
        sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
        sci.EntryPoint = "main";
        sci.HLSLVersion = { 5, 0 };


        const auto devType = m_Device->GetDeviceInfo().Type;
        const bool useCombined =
            (devType == RENDER_DEVICE_TYPE_GL || devType == RENDER_DEVICE_TYPE_GLES);

        sci.Desc.UseCombinedTextureSamplers = useCombined;
        sci.Desc.CombinedSamplerSuffix = "Sampler";

        // VS (fullscreen triangle)
        sci.Desc.Name = "Lighting VS";
        sci.Desc.ShaderType = SHADER_TYPE_VERTEX;
        sci.Source = kLightingVS;
        m_Device->CreateShader(sci, &vs);
        if (!vs) { OutputDebugStringA("[Lighting] VS creation failed.\n"); return; }

        // PS (Lambert à partir du GBuffer)
        sci.Desc.Name = "Lighting PS (Lambert)";
        sci.Desc.ShaderType = SHADER_TYPE_PIXEL;
        sci.Source = kLightingPSLambert; // <- le PS fourni précédemment
        m_Device->CreateShader(sci, &ps);
        if (!ps) { OutputDebugStringA("[Lighting] PS creation failed.\n"); return; }

        // --- PSO ---
        GraphicsPipelineStateCreateInfo psoCI{};
        psoCI.PSODesc.Name = "Deferred Lighting PSO (Lambert)";
        psoCI.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

        const auto& sc = m_Swap->GetDesc();
        psoCI.GraphicsPipeline.NumRenderTargets = 1;
        psoCI.GraphicsPipeline.RTVFormats[0] = sc.ColorBufferFormat;
        psoCI.GraphicsPipeline.DSVFormat = TEX_FORMAT_UNKNOWN; // pas de depth en composition
        psoCI.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // 3 sommets
        psoCI.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;
        psoCI.GraphicsPipeline.DepthStencilDesc.DepthEnable = False;
        psoCI.GraphicsPipeline.InputLayout = { nullptr, 0 }; // pas de VB

        // Variables dynamiques (t0,t1,t2)
        const ShaderResourceVariableDesc Vars[] = {
            {SHADER_TYPE_PIXEL, "g_PosNorm", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
            {SHADER_TYPE_PIXEL, "g_Color",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
            {SHADER_TYPE_PIXEL, "g_ID",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        };

        // Sampler immuable pour g_Color (les textures UINT sont lues avec Load -> pas de sampler)
        SamplerDesc samp{};
        samp.MinFilter = FILTER_TYPE_LINEAR;
        samp.MagFilter = FILTER_TYPE_LINEAR;
        samp.MipFilter = FILTER_TYPE_LINEAR;
        samp.AddressU = TEXTURE_ADDRESS_CLAMP;
        samp.AddressV = TEXTURE_ADDRESS_CLAMP;
        samp.AddressW = TEXTURE_ADDRESS_CLAMP;

        const ImmutableSamplerDesc ImmSamplers[] = {
            {SHADER_TYPE_PIXEL, "g_ColorSampler", samp},
        };

        psoCI.PSODesc.ResourceLayout.NumVariables = static_cast<Uint32>(std::size(Vars));
        psoCI.PSODesc.ResourceLayout.Variables = Vars;
        psoCI.PSODesc.ResourceLayout.NumImmutableSamplers = static_cast<Uint32>(std::size(ImmSamplers));
        psoCI.PSODesc.ResourceLayout.ImmutableSamplers = ImmSamplers;

        psoCI.pVS = vs;
        psoCI.pPS = ps;

        m_Device->CreateGraphicsPipelineState(psoCI, &m_LightingPSO);
        if (!m_LightingPSO) { OutputDebugStringA("[Lighting] CreateGraphicsPipelineState failed.\n"); return; }

        m_LightingPSO->CreateShaderResourceBinding(&m_LightingSRB, true);
        if (!m_LightingSRB) { OutputDebugStringA("[Lighting] CreateShaderResourceBinding failed.\n"); return; }
    }








    void DiligentRenderer3::DrawFullscreenFromSRV(ITextureView* srv)
    {
        if (!srv || !m_PostPSO)
            return;

        m_Ctx->SetPipelineState(m_PostPSO);

        if (!m_PostSRB)
        {
            m_PostPSO->CreateShaderResourceBinding(&m_PostSRB, true);
            m_PostSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_InputTex")->Set(srv);
        }

        m_Ctx->CommitShaderResources(m_PostSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        ITextureView* RTVs[] = { m_FinalRTV };
        m_Ctx->SetRenderTargets(1, RTVs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        DrawAttribs drawAttrs;
        drawAttrs.NumVertices = 3; // fullscreen triangle
        drawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
        m_Ctx->Draw(drawAttrs);

        // Maintenant, blit vers backbuffer
        auto* backRTV = m_Swap->GetCurrentBackBufferRTV();
        m_Ctx->SetRenderTargets(1, &backRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Ici tu peux soit redessiner un fullscreen quad, soit faire un CopyTexture
        // Pour simplifier :

        CopyTextureAttribs copyAttrs;

        // Source
        copyAttrs.pSrcTexture = m_FinalTex;
        copyAttrs.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;

        // Destination
        copyAttrs.pDstTexture = m_Swap->GetCurrentBackBufferRTV()->GetTexture();
        copyAttrs.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;

        // Options
        copyAttrs.SrcSlice = 0;
        copyAttrs.DstSlice = 0;
        copyAttrs.SrcMipLevel = 0;
        copyAttrs.DstMipLevel = 0;

        m_Ctx->CopyTexture(copyAttrs);
    }

} // namespace rvtx::dil
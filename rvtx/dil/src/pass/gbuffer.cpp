#include "rvtx/dil/pass/gbuffer.hpp"

#include <Graphics/GraphicsEngine/interface/RenderDevice.h>
#include <Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <Graphics/GraphicsEngine/interface/Texture.h>
#include <Graphics/GraphicsEngine/interface/TextureView.h>

using namespace Diligent;

namespace rvtx::dil
{
    GBufferPass::GBufferPass(IRenderDevice* pDevice, Uint32 width, Uint32 height)
    {
        CreateBuffers(pDevice, width, height);
    }

    void GBufferPass::CreateBuffers(IRenderDevice* pDevice, Uint32 width, Uint32 height)
    {
        m_Width = width;
        m_Height = height;

        // Positions + Normales (compressées) : RGBA32_UINT
        TextureDesc posNormDesc{};
        posNormDesc.Name = "GBuffer Positions+Normals";
        posNormDesc.Type = RESOURCE_DIM_TEX_2D;
        posNormDesc.Width = width;
        posNormDesc.Height = height;
        posNormDesc.MipLevels = 1;
        posNormDesc.ArraySize = 1;
        posNormDesc.SampleCount = 1;
        posNormDesc.Usage = USAGE_DEFAULT;
        posNormDesc.Format = TEX_FORMAT_RGBA32_UINT;
        posNormDesc.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
        pDevice->CreateTexture(posNormDesc, nullptr, &m_PosNormTex);
        m_PosNormRTV = m_PosNormTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_PosNormSRV = m_PosNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        // Couleurs / matériaux : RGBA16_FLOAT
        TextureDesc colorDesc = posNormDesc;
        colorDesc.Name = "GBuffer Colors";
        colorDesc.Format = TEX_FORMAT_RGBA16_FLOAT;
        pDevice->CreateTexture(colorDesc, nullptr, &m_ColorTex);
        m_ColorRTV = m_ColorTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_ColorSRV = m_ColorTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        // IDs : R32_UINT
        TextureDesc idDesc = posNormDesc;
        idDesc.Name = "GBuffer IDs";
        idDesc.Format = TEX_FORMAT_R32_UINT;
        pDevice->CreateTexture(idDesc, nullptr, &m_IdTex);
        m_IdRTV = m_IdTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
        m_IdSRV = m_IdTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

        // Depth : D32_FLOAT (SRV activé)
        TextureDesc depthDesc = posNormDesc;
        depthDesc.Name = "GBuffer Depth";
        depthDesc.Format = TEX_FORMAT_D32_FLOAT;
        depthDesc.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
        pDevice->CreateTexture(depthDesc, nullptr, &m_DepthTex);
        m_DepthDSV = m_DepthTex->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
        m_DepthSRV = m_DepthTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    }

    void GBufferPass::Resize(IRenderDevice* pDevice, Uint32 width, Uint32 height)
    {
        if (width == m_Width && height == m_Height) return;
        CreateBuffers(pDevice, width, height);
    }

    void GBufferPass::BeginPass(IDeviceContext* pCtx)
    {
        Viewport vp{ 0.f,0.f,(Float32)m_Width,(Float32)m_Height,0.f,1.f };
        
        
        pCtx->SetViewports(1, &vp, m_Width, m_Height);

        ITextureView* RTVs[] = { m_PosNormRTV, m_ColorRTV, m_IdRTV };
        pCtx->SetRenderTargets((Uint32)std::size(RTVs), RTVs, m_DepthDSV,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        const float Z[4] = { 0,0,0,0 };
        const float W[4] = { 1,1,1,1 };

        pCtx->ClearRenderTarget(m_PosNormRTV, Z, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        pCtx->ClearRenderTarget(m_IdRTV, Z, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        pCtx->ClearRenderTarget(m_ColorRTV, W, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        pCtx->ClearDepthStencil(m_DepthDSV, CLEAR_DEPTH_FLAG, 1.0f, 0,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        pCtx->Flush();
    }

    void GBufferPass::EndPass(IDeviceContext* pCtx)
    {
        // Débinde proprement (optionnel)
        pCtx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    void GBufferPass::Render(IDeviceContext* pCtx, GeometryDraw geometryDraw)
    {
        BeginPass(pCtx);
        if (geometryDraw) geometryDraw();
        EndPass(pCtx);
    }

} // namespace rvtx::dil

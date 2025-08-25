#ifndef RVTX_DIL_GBUFFER_PASS_HPP
#define RVTX_DIL_GBUFFER_PASS_HPP

#include <functional>
#include <RenderDevice.h>
#include <DeviceContext.h>
#include <Texture.h>
#include <TextureView.h>
#include <RefCntAutoPtr.hpp>

namespace rvtx::dil
{
    using namespace Diligent;
    using GeometryDraw = std::function<void()>;

    class GBufferPass
    {
    public:
        GBufferPass() = default;
        GBufferPass(IRenderDevice* pDevice, Uint32 width, Uint32 height);

        GBufferPass(const GBufferPass&) = delete;
        GBufferPass& operator=(const GBufferPass&) = delete;

        GBufferPass(GBufferPass&&) noexcept;
        GBufferPass& operator=(GBufferPass&&) noexcept;

        ~GBufferPass() = default;

        // SRV pour passes suivantes
        ITextureView* GetGeometrySRV() const { return m_PosNormSRV; }
        ITextureView* GetMaterialSRV() const { return m_ColorSRV; }
        ITextureView* GetIdSRV() const { return m_IdSRV; }
        ITextureView* GetDepthSRV() const { return m_DepthSRV; }

        // Démarrer/terminer la passe GBuffer
        void BeginPass(IDeviceContext* pCtx);
        void EndPass(IDeviceContext* pCtx);

        void Resize(IRenderDevice* pDevice, Uint32 width, Uint32 height);
        void Render(IDeviceContext* pCtx, GeometryDraw geometryDraw);

    private:
        void CreateBuffers(IRenderDevice* pDevice, Uint32 width, Uint32 height);

        RefCntAutoPtr<ITexture> m_PosNormTex;
        RefCntAutoPtr<ITexture> m_ColorTex;
        RefCntAutoPtr<ITexture> m_IdTex;
        RefCntAutoPtr<ITexture> m_DepthTex;

        // Views écriture (RTV/DSV)
        RefCntAutoPtr<ITextureView> m_PosNormRTV;
        RefCntAutoPtr<ITextureView> m_ColorRTV;
        RefCntAutoPtr<ITextureView> m_IdRTV;
        RefCntAutoPtr<ITextureView> m_DepthDSV;

        // Views lecture (SRV)
        RefCntAutoPtr<ITextureView> m_PosNormSRV;
        RefCntAutoPtr<ITextureView> m_ColorSRV;
        RefCntAutoPtr<ITextureView> m_IdSRV;
        RefCntAutoPtr<ITextureView> m_DepthSRV;

        Uint32 m_Width = 0;
        Uint32 m_Height = 0;

    };
} // namespace rvtx::dil

#endif // RVTX_DIL_GBUFFER_PASS_HPP

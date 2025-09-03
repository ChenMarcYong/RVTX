// DiligentRenderer.hpp
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include <Graphics/GraphicsEngine/interface/RenderDevice.h>
#include <Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <Graphics/GraphicsEngine/interface/SwapChain.h>
#include <Graphics/GraphicsEngine/interface/PipelineState.h>
#include <Graphics/GraphicsEngine/interface/ShaderResourceBinding.h>
#include <Graphics/GraphicsEngine/interface/Shader.h>
#include <Graphics/GraphicsEngine/interface/Texture.h>
#include <Graphics/GraphicsEngine/interface/GraphicsTypes.h>

#include "rvtx/system/scene.hpp"
#include "rvtx/system/camera.hpp"

#include "rvtx/dil/geometry/handler.hpp"
#include "rvtx/dil/pass/gbuffer.hpp"
#include "rvtx/dil/pass/post_process.hpp"

#ifndef COUNT_OF
#  define COUNT_OF(arr) static_cast<Diligent::Uint32>(sizeof(arr)/sizeof((arr)[0]))
#endif

namespace rvtx::dil
{
    class DiligentRenderer
    {
    public:
        DiligentRenderer() = default;
        DiligentRenderer(Diligent::IRenderDevice* dev,
            Diligent::IDeviceContext* ctx,
            Diligent::ISwapChain* swap,
            rvtx::dil::PipelineManager& pipeline,
            uint32_t width,
            uint32_t height);

        DiligentRenderer(const DiligentRenderer&) = delete;
        DiligentRenderer& operator=(const DiligentRenderer&) = delete;

        DiligentRenderer(DiligentRenderer&&) noexcept = default;
        DiligentRenderer& operator=(DiligentRenderer&&) noexcept = default;

        ~DiligentRenderer() = default;

        // --- comme ton Renderer OpenGL ---
        inline void enableUI(bool enable) { m_EnableUI = enable; }
        inline void setGeometry(std::unique_ptr<rvtx::dil::GeometryHandler>&& g) { m_Geometry = std::move(g); }

        // Expose la RT finale
        inline Diligent::ITextureView* GetFinalRTV() const { return m_FinalRTV; }
        inline Diligent::ITextureView* GetFinalSRV() const { return m_FinalSRV; }

        void Resize(uint32_t width, uint32_t height);

        // Render 
        void render(const rvtx::Camera& cam,
            const rvtx::Scene& scene,
            const std::function<void()>& updateUI = [] {});


        void SetClearColor(const Diligent::float4& c) { m_ClearColor = c; }
        const Diligent::float4& GetClearColor() const { return m_ClearColor; }

    private:
        void CreateTargets();
        void CreatePostPresentPipeline();
        void DrawFullscreenFromSRV(Diligent::ITextureView* srv);
        void CreateLightingPipeline();

    private:
        // Backend Diligent
        Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  m_Device;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_Ctx;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>     m_Swap;

        uint32_t m_Width = 0;
        uint32_t m_Height = 0;

        bool m_EnableUI = true;

        // --- Nouveau : GBuffer  ---
        std::unique_ptr<rvtx::dil::GBufferPass> m_GBuffer;
        std::unique_ptr<rvtx::dil::PostProcessPassDiligent> m_postProcessPass;
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_LightingPSO;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_LightingSRB;

        // --- Render target finale ---
        Diligent::RefCntAutoPtr<Diligent::ITexture>     m_FinalTex;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalRTV;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalSRV;



        // --- PSO / SRB pour postprocess ---
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_PostPSO;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PostSRB;

        // Géométrie active
        std::unique_ptr<rvtx::dil::GeometryHandler> m_Geometry;

        Diligent::float4 m_ClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };


        rvtx::dil::PipelineManager* m_Pipeline = nullptr;
    }; 
} // namespace rvtx::dil

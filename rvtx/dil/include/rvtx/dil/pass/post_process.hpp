#ifndef RVTX_GL_PASS_POST_PROCESS_2_HPP
#define RVTX_GL_PASS_POST_PROCESS_2_HPP

#include <cstdint>
#include <glm/vec3.hpp>

#include <RefCntAutoPtr.hpp>
#include <GraphicsTypes.h>


#include "rvtx/dil/pass/pass.hpp"
#include "rvtx/dil/utils/pipeline_manager.hpp"

namespace rvtx
{
    struct Camera;
}


namespace Diligent {
    class IRenderDevice;
    class IDeviceContext;
    class ITexture;
    class ITextureView;
    class IPipelineState;
    class IShaderResourceBinding;
    class IBuffer;
}

namespace rvtx::dil
{
    
    class PipelineManager;
    class PipelineEntry;



    class LinearizeDepthPostProcessDiligent : public Pass
    {
        public:
            
            

            LinearizeDepthPostProcessDiligent() = default;
            LinearizeDepthPostProcessDiligent(uint32_t width, uint32_t height, PipelineManager& manager);
    
            LinearizeDepthPostProcessDiligent(const LinearizeDepthPostProcessDiligent&) = delete;
            LinearizeDepthPostProcessDiligent& operator=(const LinearizeDepthPostProcessDiligent&) = delete;

            LinearizeDepthPostProcessDiligent(LinearizeDepthPostProcessDiligent&&) noexcept;
            LinearizeDepthPostProcessDiligent& operator=(LinearizeDepthPostProcessDiligent&&) noexcept;

            //~LinearizeDepthPostProcessDiligent() override;
            ~LinearizeDepthPostProcessDiligent() override = default;
    
            void setInputTexture(Diligent::ITextureView* pSRV);
            Diligent::ITextureView* getTexture() const;
            
            void resize(Diligent::IRenderDevice* pDevice, uint32_t width, uint32_t height) override;
            void render(const Camera& camera);


            void Execute(
                Diligent::IDeviceContext* ctx,
                const rvtx::Camera& camera,
                Diligent::ITextureView* depthSRV);

            //void Execute(Diligent::ITextureView* pDepthSRV, const CameraCBData& camData);

            void render(Diligent::IDeviceContext* pCtx, const rvtx::Camera& camera);

            PipelineManager& getManager() { return *m_Manager; };
            Diligent::ITextureView* getSRV() const { return m_OutputSRV; }

            Diligent::RefCntAutoPtr<Diligent::ITextureView> m_OutputSRV;
            //Diligent::RefCntAutoPtr<Diligent::IBuffer> pDebugCB;
            //DebugCBData            m_pDebugCB{ {0.0f, 0.06f}, 0.7f, 0u };
        private :
            void createTarget(Diligent::IRenderDevice* pDevice, uint32_t w, uint32_t h);
            void createPSOIfNeeded(Diligent::IRenderDevice* pDevice);
            void updateConstants(Diligent::IDeviceContext* pCtx, const rvtx::Camera& camera);

            Diligent::RefCntAutoPtr<Diligent::ITexture>     m_Output;
            Diligent::RefCntAutoPtr<Diligent::ITextureView> m_OutputRTV;
            

            // Pipeline / bindings
            Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_PSO;
            Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_SRB;
            Diligent::RefCntAutoPtr<Diligent::IBuffer>                m_pCameraCB; // near/far/perspective

            // Optionnel: référence vers un gestionnaire de pipelines (si tu l’utilises)
            PipelineManager* m_Manager = nullptr;
            PipelineManager::PipelineEntry* m_PipelineEntry;
            PipelineManager::PipelineEntry* m_PipelineEntryDebug;


            // Format de sortie (R16F ou R32F)
            Diligent::TEXTURE_FORMAT m_OutputFormat = Diligent::TEX_FORMAT_R32_FLOAT;
            bool useDebug = false;


            Diligent::RefCntAutoPtr<Diligent::IBuffer> m_pDebugCB{};
            struct DebugCBData { float Window[2]; float Gamma; Diligent::Uint32 Invert; };
            DebugCBData m_DebugCBData{ {0.0f, 0.03f}, 0.6f, 0u };
    };



    class SSAOPostProcessDiligent : public Pass
    {
    public:
        SSAOPostProcessDiligent() = default;
        SSAOPostProcessDiligent(uint32_t width, uint32_t height, PipelineManager& manager);

        SSAOPostProcessDiligent(const SSAOPostProcessDiligent&) = delete;
        SSAOPostProcessDiligent& operator=(const SSAOPostProcessDiligent&) = delete;
        SSAOPostProcessDiligent(SSAOPostProcessDiligent&&) noexcept = default;
        SSAOPostProcessDiligent& operator=(SSAOPostProcessDiligent&&) noexcept = default;

        ~SSAOPostProcessDiligent() override = default;

        // Entrées
        //void setDepthTexture(Diligent::ITextureView* pLinearDepthSRV);     // texture depth linéarisée (t0)
        //void setGeometricTexture(Diligent::ITextureView* pGeometrySRV);    // normals/pos compressés (t1)

        // Sortie
        Diligent::ITextureView* getTexture() const { return m_OutputSRV; }

        // Ressources
        void resize(Diligent::IRenderDevice* pDevice, uint32_t width, uint32_t height) override;
        void render(const rvtx::Camera& camera);
        void renderToBackBuffer(const rvtx::Camera& camera);
        // Réglages
        //void setKernelSize(uint32_t k) { m_KernelSize = k; }
        //void setNoiseTexSize(uint32_t n) { m_NoiseTextureSize = n; }
        //void setAOIntensity(float i) { m_AOIntensity = i; }

        void setViewPosNormalSRV(Diligent::ITextureView* srv) { m_ViewPosNormalSRV = srv; }
        void setLinearDepthSRV(Diligent::ITextureView* srv) { m_LinearDepthSRV = srv; }


        PipelineManager& getManager() { return *m_Manager; };
        void createCBuffers(Diligent::IRenderDevice* pDevice);

    private:
        // Helpers d’implémentation (dans le .cpp)
        void createTarget(Diligent::IRenderDevice* pDevice, uint32_t w, uint32_t h);
        void createNoiseTexture(Diligent::IRenderDevice* pDevice, uint32_t size);
        //void createPSOIfNeeded(Diligent::IRenderDevice* pDevice);
        //void updateConstants(Diligent::IDeviceContext* pCtx, const rvtx::Camera& camera);
        void generateKernel(); // (re)génère m_AOKernel selon m_KernelSize

        // Entrées (non-owning)
        Diligent::ITextureView* m_LinearDepthSRV = nullptr; // t0
        Diligent::ITextureView* m_ViewPosNormalSRV = nullptr; // t1 (normals/pos/packed)

        // Sortie AO (owning)
        Diligent::RefCntAutoPtr<Diligent::ITexture>     m_Output;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> m_OutputRTV;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> m_OutputSRV;

        // Bruit (owning)
        Diligent::RefCntAutoPtr<Diligent::ITexture>     m_Noise;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> m_NoiseSRV;

        // Pipeline / bindings
        Diligent::RefCntAutoPtr<Diligent::IPipelineState>         m_PSO;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_SRB;

        // Constantes (proj, kernel, intensité, params)
        Diligent::RefCntAutoPtr<Diligent::IBuffer> m_CB;  // matrices/paramètres scalaires
        Diligent::RefCntAutoPtr<Diligent::IBuffer> m_CBKernel; // tableau d’échantillons
        Diligent::RefCntAutoPtr<Diligent::IBuffer> m_CBDebug;
        // Réglages SSAO
        uint32_t               m_KernelSize = 32;
        uint32_t               m_NoiseTextureSize = 32;
        float                  m_AOIntensity = 1.8f;
        std::vector<glm::vec3> m_AOKernel;

        // Format de sortie (R8_UNORM ou R16_FLOAT)
        Diligent::TEXTURE_FORMAT m_OutputFormat = Diligent::TEX_FORMAT_R8_UNORM;

        // Optionnel : accès au gestionnaire
        PipelineManager* m_Manager = nullptr;
    };



    class PostProcessPassDiligent : public Pass
    {
    public:



        PostProcessPassDiligent() = default;
        PostProcessPassDiligent(uint32_t width, uint32_t height, PipelineManager& manager);

        PostProcessPassDiligent(const PostProcessPassDiligent&) = delete;
        PostProcessPassDiligent& operator=(const PostProcessPassDiligent&) = delete;

        PostProcessPassDiligent(PostProcessPassDiligent&&) noexcept = default;
        PostProcessPassDiligent& operator=(PostProcessPassDiligent&&) noexcept = default;

        ~PostProcessPassDiligent() override = default;

        void resize(Diligent::IRenderDevice* device, uint32_t w, uint32_t h) override;

        inline bool getEnableAO() const { return m_enableAO; }
        inline void setEnableAO(bool enable) { m_enableAO = enable; }
        inline void toggleEnableAO() { m_enableAO = !m_enableAO; }

        bool ok;
        LinearizeDepthPostProcessDiligent m_linearizeDepth;
        SSAOPostProcessDiligent m_ssao;
    protected:
        PipelineManager* m_PipelineManager = nullptr;
        bool m_enableAO = true;


    private:
        
    };

}


#endif



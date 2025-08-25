#include "diligent_pipeline_deffered_ball.hpp"
#include "data/data.cpp"


#include <rvtx/dil/geometry/sphere_holder.hpp>
#include <rvtx/dil/geometry/ball_and_stick_holder.hpp>
#include <rvtx/molecule/loader.hpp>

#include "rvtx/dil/utils/pipeline_manager.hpp"

#include <Graphics/GraphicsEngine/interface/Shader.h>

#include <filesystem>
#include <windows.h>
#include <string>


#include <fstream>
#include <sstream>
#include <stdexcept>

#include "MapHelper.hpp"


//rvtx

#include <entt/entity/registry.hpp>
#include <fmt/chrono.h> // Used to handle 'std::tm' in fmt::format
#include <rvtx/core/logger.hpp>
#include <rvtx/core/time.hpp>
#include <rvtx/molecule/molecule.hpp>
#include <rvtx/system/camera.hpp>
#include <rvtx/system/name.hpp>
//#include <rvtx/system/scene.hpp>
#include <rvtx/system/scene_descriptor.hpp>
#include <rvtx/system/transform.hpp>
#include <rvtx/molecule/color.hpp>

#include <array>

#include "InputController.hpp"


#include <unordered_map>
#include <cstdint>

#include <Windows.h>
#include <filesystem>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using namespace Diligent;

namespace Diligent
{
    // Cette fonction doit exister et créer ton sample
    SampleBase* CreateSample()
    {
        return new rvtx::dil::DiligentDeffered();
    }
}

namespace rvtx::dil
{

    void DiligentDeffered::InitSnapshotDir()
    {
#ifdef _WIN32
        wchar_t exePathW[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exePathW, MAX_PATH);
        std::filesystem::path exePath(exePathW);
        m_SnapshotDir = exePath.parent_path() / "snapshots";
#else
        m_SnapshotDir = std::filesystem::current_path() / "snapshots";
#endif

        std::error_code ec;
        std::filesystem::create_directories(m_SnapshotDir, ec); // ok si déjà existant
    }


    void DiligentDeffered::TakeScreenshot()
    {
        using namespace Diligent;

        ITextureView* pRTV = m_pSwapChain ? m_pSwapChain->GetCurrentBackBufferRTV() : nullptr;
        if (!pRTV) return;

        ITexture* pSrcTex = pRTV->GetTexture();
        if (!pSrcTex) return;

        const auto& SrcDesc = pSrcTex->GetDesc();

        TextureDesc StagingDesc = SrcDesc;
        StagingDesc.BindFlags = BIND_NONE;
        StagingDesc.MipLevels = 1;
        StagingDesc.ArraySize = 1;
        StagingDesc.Usage = USAGE_STAGING;
        StagingDesc.CPUAccessFlags = CPU_ACCESS_READ;

        RefCntAutoPtr<ITexture> pStaging;
        m_pDevice->CreateTexture(StagingDesc, nullptr, &pStaging);
        if (!pStaging) return;

        CopyTextureAttribs cta;
        cta.pSrcTexture = pSrcTex;
        cta.pDstTexture = pStaging;
        cta.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        cta.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
        m_pImmediateContext->CopyTexture(cta);

        MappedTextureSubresource M{};
        m_pImmediateContext->MapTextureSubresource(pStaging, 0, 0, MAP_READ, MAP_FLAG_NONE, nullptr, M);
        if (!M.pData) return;

        const int w = static_cast<int>(SrcDesc.Width);
        const int h = static_cast<int>(SrcDesc.Height);
        const int comp = 4;

        const bool isBGRA = (SrcDesc.Format == TEX_FORMAT_BGRA8_UNORM || SrcDesc.Format == TEX_FORMAT_BGRA8_UNORM_SRGB);

        std::vector<unsigned char> pixels(size_t(w) * h * comp);
        for (int y = 0; y < h; ++y) {
            auto* dst = pixels.data() + size_t(y) * w * comp;
            auto* src = static_cast<const unsigned char*>(M.pData) + size_t(y) * M.Stride;
            memcpy(dst, src, size_t(w) * comp);
            if (isBGRA) {
                for (int x = 0; x < w; ++x) std::swap(dst[x * 4 + 0], dst[x * 4 + 2]);
            }
        }
        m_pImmediateContext->UnmapTextureSubresource(pStaging, 0, 0);

        // Nom daté
        SYSTEMTIME st{}; GetLocalTime(&st);
        char fname[128];
        sprintf_s(fname, "snapshot_%04d-%02d-%02d_%02d-%02d-%02d.png",
            st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

        // Chemin final : .../bin/snapshots/snapshot_*.png
        std::filesystem::path outPath = m_SnapshotDir / fname;

        // Écriture PNG (stb_image_write)
        stbi_flip_vertically_on_write(1);
        const std::string outUtf8 = outPath.string(); // ok si ASCII; pour noms non ASCII, convertir en UTF-8 proprement
        stbi_write_png(outUtf8.c_str(), w, h, comp, pixels.data(), w * comp);
    }



    void DiligentDeffered::WindowResize(Diligent::Uint32 Width, Diligent::Uint32 Height)
    {
        SampleBase::WindowResize(Width, Height);
        if (auto* dili = dynamic_cast<DiligentInputAdapter*>(m_InputAdapter.get()))
            dili->OnResize();

        if (m_RvtxCamera)
            m_RvtxCamera->viewport = { Width, Height };

        SetupViewPort();
    }


    std::vector<Molecule> DiligentDeffered::loadAllMoleculesFromScene2(
        const std::filesystem::path& sceneJsonPath,
        rvtx::CameraDescriptor& cd,
        rvtx::Camera& camera)
    {
        std::vector<Molecule> molecules;

        const SceneDescriptor sceneDesc = parse(sceneJsonPath);
        const auto baseDir = std::filesystem::absolute(sceneJsonPath).parent_path();

        for (std::size_t i = 0; i < sceneDesc.entities.size(); ++i)
        {
            const auto& e = sceneDesc.entities[i];
            const entt::handle entity = scene.createEntity("sphere");
            auto& transform = entity.emplace<rvtx::Transform>(e.transform);

            if (e.type == rvtx::EntityDescriptor::Molecule)
            {
                // Résolution absolue
                std::filesystem::path absPath = baseDir / e.path;
                std::error_code ec;
                absPath = std::filesystem::weakly_canonical(absPath, ec);
                bool ok = std::filesystem::exists(absPath);

                char dbg[512];
                sprintf_s(dbg,
                    "[LOAD] entity %zu type=%d path='%s' exists=%d\n",
                    i, int(e.type), absPath.string().c_str(), ok ? 1 : 0);
                OutputDebugStringA(dbg);

                try
                {
                    // IMPORTANT: passer le chemin ABSOLU ici
                    Molecule m = rvtx::load(absPath);
                    molecules.push_back(m);

                    auto& molecule = entity.emplace<rvtx::Molecule>(m);
                    molecule.aabb.attachTransform(&transform);

                    auto sphereHolder = rvtx::dil::SphereHolder2::getMolecule(m_pDevice, molecule);
                    entity.emplace<rvtx::dil::SphereHolder2>(std::move(sphereHolder));

                    if (cd.targetEntity == i)
                        camera.target = rvtx::Camera::Target(molecule.getAabb());

                    OutputDebugStringA("[LOAD] added SphereHolder2\n");
                }
                catch (const std::exception& ex)
                {
                    std::string msg = std::string("[LOAD][EXCEPTION] ") + ex.what() + "\n";
                    OutputDebugStringA(msg.c_str());
                }
            }
        }
        return molecules;
    }



    // ==========================================================
    // Initialisation
    // ==========================================================

    void DiligentDeffered::Initialize(const Diligent::SampleInitInfo& InitInfo)
    {
        // Initialisation de la base
        SampleBase::Initialize(InitInfo);

        InitSnapshotDir();

        SetupViewPort();
        // Setup caméra et matrices


        glm::uvec2                 _viewport = rd.useWindowViewport ? wd.viewport : rd.viewport;

        rvtx::CameraDescriptor& cd = sceneDescriptor.cameraDescriptor;
        entt::handle             cameraEntity = scene.createEntity("Main Camera");
        rvtx::Transform& cameraTransform = cameraEntity.emplace<rvtx::Transform>(cd.transform);
        rvtx::Camera& camera = cameraEntity.emplace<rvtx::Camera>(
            cameraTransform, _viewport, cd.target, cd.projectionType, glm::radians(cd.fov));
        m_CamForwarder = std::make_unique<rvtx::ControllerForwarder>();
        auto& camCtrl = m_CamForwarder->add<rvtx::CameraController>(cameraEntity);
        camCtrl.setType(rvtx::CameraController::Type::Trackball);
        SetRvtxCamera(camera);


        auto& ic = GetInputController();
        m_InputAdapter = std::make_unique<DiligentInputAdapter>(ic, m_pSwapChain);

        
        //m_Gbuffer = std::make_unique<rvtx::dil::GBufferPass>(m_pDevice, wd.width, wd.height);
        
        std::vector<Molecule> molecules = loadAllMoleculesFromScene2("C:/M2 ISICG/Projet M2/rvtx/final/rVTX/examples_diligent/diligent_pipeline_deffered_ball/src/scene_2AGA.json", cd, *m_RvtxCamera);
        
        {
            auto nAlive = scene.registry.alive();
            auto nHolders = scene.registry.view<rvtx::dil::SphereHolder2>().size();
            auto nTH = scene.registry.view<rvtx::Transform, rvtx::dil::SphereHolder2>().size_hint();

            char buf[256];
            sprintf_s(buf, "[SCENE] alive=%zu  holders=%zu  TH=%zu/n",
                static_cast<size_t>(nAlive),
                static_cast<size_t>(nHolders),
                static_cast<size_t>(nTH));
            OutputDebugStringA(buf);
        }




        pipeline = std::make_unique<PipelineManager>(m_pDevice, m_pEngineFactory);

        const auto& sc = m_pSwapChain->GetDesc();


        m_Renderer = std::make_unique<rvtx::dil::DiligentRenderer3>(
            m_pDevice, m_pImmediateContext, m_pSwapChain, *pipeline,
            sc.Width, sc.Height
        );



        //auto geomSphere = std::make_unique<rvtx::dil::SphereHandler>(
        //    *pipeline,                       // PipelineManager
        //    m_pDevice, m_pImmediateContext, m_pSwapChain, m_pEngineFactory
        //);


        //auto geomBall = std::make_unique<rvtx::dil::BallAndStickHandler>(
        //    *pipeline,                       // PipelineManager
        //    m_pDevice, m_pImmediateContext, m_pSwapChain, m_pEngineFactory
        //);

        auto geometryForwarder = std::make_unique<rvtx::dil::GeometryForwarder>();

        geometryForwarder->add<rvtx::dil::SphereHandler>(
            *pipeline,                       // PipelineManager
            m_pDevice, m_pImmediateContext, m_pSwapChain, m_pEngineFactory
            );

        geometryForwarder->add<rvtx::dil::BallAndStickHandler>(
            *pipeline,                       // PipelineManager
            m_pDevice, m_pImmediateContext, m_pSwapChain, m_pEngineFactory
        );




        m_Renderer->setGeometry(std::move(geometryForwarder));

        char buf2[256];
        sprintf_s(buf2, "[PIPELINE] size=%zu/n",
            static_cast<size_t>(pipeline->m_pipelines.size()));

        OutputDebugStringA(buf2);
    
    }

    void DiligentDeffered::SetupViewPort()
    {
        glm::uvec2                 viewport = rd.useWindowViewport ? wd.viewport : rd.viewport;

        vp.TopLeftX = 0;
        vp.TopLeftY = 0;
        vp.Width = static_cast<float>(viewport.x);
        vp.Height = static_cast<float>(viewport.y);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;

        m_pImmediateContext->SetViewports(1, &vp, vp.Width, vp.Height);
    }

    // ==========================================================
    // Render
    // ==========================================================

    void DiligentDeffered::Update(double CurrTime, double ElapsedTime, bool DoUpdateUI)
    {
        SampleBase::Update(CurrTime, ElapsedTime, DoUpdateUI);

        // Caméra Diligent (si tu l’utilises pour d’autres passes)
        m_Camera.Update(GetInputController(), static_cast<float>(ElapsedTime));
        m_ViewMatrix = m_Camera.GetViewMatrix();
        m_ProjMatrix = m_Camera.GetProjMatrix();

        if (!m_RvtxCamera) return;

        // Garde l’aspect à jour (optionnel : tu peux le faire seulement au resize)
        const auto& sc = m_pSwapChain->GetDesc();
        m_RvtxCamera->viewport = { sc.Width, sc.Height };

        // ---- INPUT UNIFIÉ ----
        // Remplit m_Input (un seul état d’input pour tout rvtx)

        m_InputAdapter->Poll(static_cast<float>(ElapsedTime), m_Input);

        if (m_Input.isKeyDown(rvtx::Key::F7))
            TakeScreenshot();

        // Le contrôleur caméra rvtx consomme m_Input
        if (m_CamForwarder)
            m_CamForwarder->update(m_Input);


    }

    void DiligentDeffered::Render()
    {



        if (!m_RvtxCamera || !m_Renderer) return;

        // Exemple : petite lambda si tu veux dessiner une UI (facultatif)
        auto drawUI = []() { /* ImGui via DiligentTools si tu l’as branché */ };

        auto view = scene.registry.view<rvtx::Transform, rvtx::dil::SphereHolder2>();
        char buf[128];
        sprintf_s(buf, "[RENDER] spheres to draw = %zu/n", (size_t)view.size_hint());
        OutputDebugStringA(buf);


        m_Renderer->Render(*m_RvtxCamera, scene, drawUI);


        
    }

    Diligent::DesiredApplicationSettings DiligentDeffered::GetDesiredApplicationSettings(bool IsInitialization)
    {

        sceneDescriptor = rvtx::parse("C:/M2 ISICG/Projet M2/rvtx/final/rVTX/examples_diligent/diligent_pipeline_deffered_ball/src/scene_2AGA.json");
        wd = sceneDescriptor.windowDescriptor;
        rd = sceneDescriptor.rendererDescriptor;

    
        DesiredApplicationSettings settings;
        settings.SetWindowWidth(wd.width)
            .SetWindowHeight(wd.height)
            .SetVSync(true)
            .SetShowUI(true);
        return settings;
    }





} // namespace rvtx::dil

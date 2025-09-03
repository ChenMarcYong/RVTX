#include "diligent_pipeline_deffered_ball.hpp"


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
#include <fmt/chrono.h>
#include <rvtx/core/logger.hpp>
#include <rvtx/core/time.hpp>
#include <rvtx/molecule/molecule.hpp>
#include <rvtx/system/camera.hpp>
#include <rvtx/system/name.hpp>
#include <rvtx/system/scene_descriptor.hpp>
#include <rvtx/system/transform.hpp>
#include <rvtx/molecule/color.hpp>

#include <array>

#include "InputController.hpp"


#include <unordered_map>
#include <cstdint>

#include "stb_image_write.h"


using namespace Diligent;

namespace Diligent
{
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
        std::filesystem::create_directories(m_SnapshotDir, ec);
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

        SYSTEMTIME st{}; GetLocalTime(&st);
        char fname[128];
        sprintf_s(fname, "snapshot_%04d-%02d-%02d_%02d-%02d-%02d.png",
            st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

        std::filesystem::path outPath = m_SnapshotDir / fname;

        stbi_flip_vertically_on_write(1);
        const std::string outUtf8 = outPath.string(); 
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


    std::vector<Molecule> DiligentDeffered::loadAllMoleculesFromScene(
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

                try
                {
                    Molecule m = rvtx::load(absPath);
                    molecules.push_back(m);

                    auto& molecule = entity.emplace<rvtx::Molecule>(m);
                    molecule.aabb.attachTransform(&transform);

                    auto sphereHolder = rvtx::dil::SphereHolder::getMolecule(m_pDevice, molecule);
                    entity.emplace<rvtx::dil::SphereHolder>(std::move(sphereHolder));

                    if (cd.targetEntity == i)
                        camera.target = rvtx::Camera::Target(molecule.getAabb());

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
        SampleBase::Initialize(InitInfo);

        InitSnapshotDir();

        SetupViewPort();


        glm::uvec2 _viewport = rd.useWindowViewport ? wd.viewport : rd.viewport;

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

        
        
        std::vector<Molecule> molecules = loadAllMoleculesFromScene("data/scene_2AGA.json", cd, *m_RvtxCamera);
        pipeline = std::make_unique<PipelineManager>(m_pDevice, m_pEngineFactory);

        const auto& sc = m_pSwapChain->GetDesc();


        m_Renderer = std::make_unique<rvtx::dil::DiligentRenderer>(
            m_pDevice, m_pImmediateContext, m_pSwapChain, *pipeline,
            sc.Width, sc.Height
        );


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

        m_Camera.Update(GetInputController(), static_cast<float>(ElapsedTime));
        m_ViewMatrix = m_Camera.GetViewMatrix();
        m_ProjMatrix = m_Camera.GetProjMatrix();

        if (!m_RvtxCamera) return;

        const auto& sc = m_pSwapChain->GetDesc();
        m_RvtxCamera->viewport = { sc.Width, sc.Height };


        m_InputAdapter->Poll(static_cast<float>(ElapsedTime), m_Input);

        if (m_Input.isKeyDown(rvtx::Key::F7))
            TakeScreenshot();


        if (m_CamForwarder)
            m_CamForwarder->update(m_Input);

        if (m_Input.windowResized)
        {
            m_RvtxCamera->viewport = m_Input.windowSize;
            m_Renderer->Resize(m_Input.windowSize.x, m_Input.windowSize.y);
        }


    }

    void DiligentDeffered::Render()
    {



        if (!m_RvtxCamera || !m_Renderer) return;


        auto drawUI = []() {};


        m_Renderer->render(*m_RvtxCamera, scene, drawUI);


        
    }

    Diligent::DesiredApplicationSettings DiligentDeffered::GetDesiredApplicationSettings(bool IsInitialization)
    {

        sceneDescriptor = rvtx::parse("data/scene_2AGA.json");
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

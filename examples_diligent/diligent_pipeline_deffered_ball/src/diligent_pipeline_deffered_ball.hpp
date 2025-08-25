/*
 *  Copyright 2019-2025 Diligent Graphics LLC
 *  Copyright 2015-2019 Egor Yusov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  In no event and under no legal theory, whether in tort (including negligence),
 *  contract, or otherwise, unless required by applicable law (such as deliberate
 *  and grossly negligent acts) or agreed to in writing, shall any Contributor be
 *  liable for any damages, including any direct, indirect, special, incidental,
 *  or consequential damages of any character arising as a result of this License or
 *  out of the use or inability to use the software (including but not limited to damages
 *  for loss of goodwill, work stoppage, computer failure or malfunction, or any and
 *  all other commercial damages or losses), even if such Contributor has been advised
 *  of the possibility of such damages.
 */


#include <rvtx/ux/camera_controller.hpp>
#include <rvtx/system/transform.hpp>
#include <rvtx/system/camera.hpp>
#include <rvtx/system/scene_descriptor.hpp>

#pragma once

#include "SampleBase.hpp"
#include "rvtx/dil/utils/pipeline_manager.hpp"

#include "rvtx/dil/geometry/sphere_handler.hpp"
#include "rvtx/dil/geometry/ball_and_stick_handler.hpp"
#include "rvtx/dil/geometry/handler.hpp"

#include "rvtx/dil/adapter/diligent_input_adapter.hpp"
#include "rvtx/dil/renderer.hpp"
#include "rvtx/dil/pass/gbuffer.hpp"

#include <filesystem>




namespace rvtx::dil
{
    class DiligentDeffered final : public Diligent::SampleBase
    {
    public:
        // Méthodes principales de SampleBase
        virtual void Initialize(const Diligent::SampleInitInfo& InitInfo) override final;
        virtual void Render() override final;
        virtual void Update(double CurrTime, double ElapsedTime, bool DoUpdateUI) override final;

        virtual Diligent::DesiredApplicationSettings GetDesiredApplicationSettings(bool IsInitialization) override final;

        void extractExtention(const std::filesystem::path& path);
        void SetupViewPort();
        void loadMolecules();

        void BuildRvtxInput(float dt);

        void SetRvtxCamera(rvtx::Camera& cam) { m_RvtxCamera = &cam; }

        void WindowResize(Diligent::Uint32 Width, Diligent::Uint32 Height) override;
        void TakeScreenshot();
        void InitSnapshotDir();

        std::vector<Molecule> loadAllMoleculesFromScene(const std::filesystem::path& sceneJsonPath, rvtx::CameraDescriptor& cd, rvtx::Camera& camera);
        std::vector<Molecule> loadAllMoleculesFromScene2(const std::filesystem::path& sceneJsonPath, rvtx::CameraDescriptor& cd, rvtx::Camera& camera);

        PipelineManager::PipelineEntry* pipelineEntry;

        rvtx::Scene scene{};
        const entt::handle entity;
        std::vector<Molecule> listMolecules;

        std::vector<Sphere> moleculeData;

        //std::unique_ptr<Camera&> camera;
        rvtx::Camera* m_RvtxCamera = nullptr;
        rvtx::ControllerForwarder forwarder{};

        rvtx::Input m_Input{};
        uint32_t    m_PrevMouseButtons = 0;
        int         m_PrevMouseX = 0, m_PrevMouseY = 0;
        bool        m_WindowResizedFlag = false;
        std::unique_ptr<rvtx::ControllerForwarder> m_CamForwarder;

        std::unique_ptr<rvtx::dil::GeometryForwarder> m_GeometryForwarder;

        rvtx::Input m_Inputs{};
        std::unique_ptr<InputAdapter> m_InputAdapter;

        std::filesystem::path m_SnapshotDir;

        std::unique_ptr<rvtx::dil::DiligentRenderer3> m_Renderer;
        std::unique_ptr<rvtx::dil::GBufferPass> m_Gbuffer;


        

    private:

    private:
        // Gestion des sphères

        

        // Buffers

        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSphereSettingsCB;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pSpheresBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresBufferView;
        Diligent::RefCntAutoPtr<Diligent::IBuffer>     m_pIDsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBufferView> m_pSpheresIdsBufferView;

        // pipelineManager

        std::unique_ptr<PipelineManager> pipeline;
        PipelineData pipelineData{ m_pDevice, m_pImmediateContext, m_pSwapChain, m_pEngineFactory};


        // Matrices
        Diligent::float4x4 m_ViewMatrix;
        Diligent::float4x4 m_ProjMatrix;

        // Caméra first person
        Diligent::FirstPersonCamera m_Camera;




        
        Scene m_scene;


        // rvtx

        rvtx::SceneDescriptor sceneDescriptor;
        rvtx::WindowDescriptor wd;
        rvtx::RendererDescriptor rd;
        Diligent::Viewport vp;

        // Molecules

        
    };

}




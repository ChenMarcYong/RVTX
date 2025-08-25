#ifndef RVTX_DIL_PIPELINE_MANAGER_HPP
#define RVTX_DIL_PIPELINE_MANAGER_HPP

#include <unordered_map>
#include <memory>
#include <filesystem>
#include <vector>
#include <string>


#include "Common/interface/RefCntAutoPtr.hpp"
#include "Graphics/GraphicsEngine/interface/Shader.h"
#include "Graphics/GraphicsEngine/interface/PipelineState.h"
#include "Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"
#include "Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "Graphics/GraphicsEngine/interface/RenderDevice.h"

#include "BasicMath.hpp"

namespace rvtx::dil
{

    struct PipelineData
    {
        Diligent::RefCntAutoPtr<Diligent::IRenderDevice>  device;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext> immediateContext;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>     swapChain;
        Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory;
    };



    class PipelineManager
    {
    public:
        struct PipelineEntry
        {
            Diligent::RefCntAutoPtr<Diligent::IPipelineState>          PSO;
            Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>  SRB;
        };

        struct CreatePipelineResult
        {
            PipelineEntry *entry;
            bool isNew;
        };

        PipelineManager(Diligent::IRenderDevice* pDevice, std::string shaderRoot);

        PipelineManager(Diligent::IRenderDevice* pDevice, Diligent::RefCntAutoPtr<Diligent::IEngineFactory> pEngineFactory);
        PipelineManager(Diligent::RefCntAutoPtr<Diligent::IRenderDevice> device,
            Diligent::RefCntAutoPtr<Diligent::IDeviceContext> context,
            Diligent::RefCntAutoPtr<Diligent::ISwapChain> swapChain,
            Diligent::RefCntAutoPtr<Diligent::IEngineFactory> engineFactory);

        PipelineManager(PipelineData& pipelineData);

        // Crée ou retourne un pipeline existant
        CreatePipelineResult create(const std::string& name,
            const std::vector<std::filesystem::path>& shaderPaths,
            Diligent::GraphicsPipelineStateCreateInfo &PSOCreateInfo,
            Diligent::ShaderResourceVariableDesc Vars[],
            size_t numVars);

        PipelineEntry* create2(const std::string& name,
            const std::vector<std::filesystem::path>& shaderPaths,
            Diligent::GraphicsPipelineStateCreateInfo& PSOCreateInfo,
            Diligent::ShaderResourceVariableDesc Vars[],
            size_t numVars);


        // Récupère un pipeline existant
        PipelineEntry* get(const std::string& name);

        Diligent::RefCntAutoPtr<Diligent::IShader> createShader(const std::filesystem::path& path);

        PipelineEntry* CreateGraphicsPipeline(Diligent::GraphicsPipelineStateCreateInfo& PSOCreateInfo);
        void AddVariablePSO(Diligent::ShaderResourceVariableDesc Vars[], size_t numVars, Diligent::GraphicsPipelineStateCreateInfo& PSOCreateInfo, Diligent::RefCntAutoPtr<Diligent::IPipelineState>& m_pPSO);  //Diligent::ShaderResourceVariableDesc Vars[] 

        void ThrowShaderError(const std::string& message);



        Diligent::IRenderDevice* m_pDevice = nullptr;
        Diligent::RefCntAutoPtr<Diligent::IDeviceContext>              m_pImmediateContex;
        Diligent::RefCntAutoPtr<Diligent::IEngineFactory>              m_pEngineFactory;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IDeviceContext>> m_pDeferredContexts;
        Diligent::RefCntAutoPtr<Diligent::ISwapChain>                  m_pSwapChain;


        PipelineData pipelineData;

        //Diligent::RefCntAutoPtr<Diligent::IPipelineState> pPSO;
        //Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> pSRB;


        uint32_t  m_NumSpheres = 0;


        std::unordered_map<std::string, PipelineEntry> m_pipelines;
    private:


        
        std::string              m_shaderRoot;
        

    };
}

#endif // RVTX_DIL_PIPELINEMANAGER_HPP

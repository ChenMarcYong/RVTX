#include "rvtx/dil/utils/pipeline_manager.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <Graphics/GraphicsEngine/interface/Shader.h>

#include <windows.h>
#include <string>

#include "MapHelper.hpp"

using namespace Diligent;

namespace rvtx::dil
{

    // Détecte le type de shader en fonction de l’extension
    static SHADER_TYPE GetShaderTypeFromExtension(const std::filesystem::path& path)
    {
        auto ext = path.extension().string();
        if (ext == ".vsh") return SHADER_TYPE_VERTEX;
        if (ext == ".psh") return SHADER_TYPE_PIXEL;
        if (ext == ".gsh") return SHADER_TYPE_GEOMETRY;
        if (ext == ".comp") return SHADER_TYPE_COMPUTE;
        return SHADER_TYPE_UNKNOWN;
    }

    PipelineManager::PipelineManager(IRenderDevice* pDevice, std::string shaderRoot)
        : m_pDevice(pDevice), m_shaderRoot(std::move(shaderRoot))
    {}

    PipelineManager::PipelineManager(IRenderDevice* pDevice, Diligent::RefCntAutoPtr<Diligent::IEngineFactory> pEngineFactory)
        : m_pDevice(pDevice), m_pEngineFactory(pEngineFactory)
    {}

    PipelineManager::PipelineManager(PipelineData &pipelineData)
        : pipelineData(pipelineData)
    {
    }

    PipelineManager::PipelineManager(RefCntAutoPtr<IRenderDevice> device,
        RefCntAutoPtr<IDeviceContext> context,
        RefCntAutoPtr<ISwapChain> swapChain,
        RefCntAutoPtr<IEngineFactory> engineFactory)
        : m_pDevice(device), m_pImmediateContex(context), m_pSwapChain(swapChain), m_pEngineFactory(engineFactory)
    {
    }

    PipelineManager::CreatePipelineResult PipelineManager::create(
        const std::string& name,
        const std::vector<std::filesystem::path>& shaderPaths,
        GraphicsPipelineStateCreateInfo &PSOCreateInfo,
        Diligent::ShaderResourceVariableDesc Vars[],
        size_t numVars)
    {


        Diligent::RefCntAutoPtr<Diligent::IPipelineState> pPSO;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> SRB;

        // Vérifier si déjà existant
        //if (m_pipelines.find(name) != m_pipelines.end())
        //{

        //    PipelineEntry entry{ m_pipelines[name].PSO, m_pipelines[name].SRB};

        //    OutputDebugStringA("PipelineEntry already exists \n");
        //    return { &entry, false };
        //}
        // Créer shaders
        RefCntAutoPtr<IShader> pVS, pPS, pGS;
        for (const auto& shaderPath : shaderPaths)
        {
            Diligent::SHADER_TYPE type = GetShaderTypeFromExtension(shaderPath);
            auto shader = createShader(shaderPath);


            switch (type)
            {
            case SHADER_TYPE_VERTEX:   
                pVS = shader; 
                OutputDebugStringA("Succesfully open vertex shader file \n");
                break;
            case SHADER_TYPE_PIXEL:
                pPS = shader; 
                OutputDebugStringA("Succesfully open pixel shader file \n");
                break;
            case SHADER_TYPE_GEOMETRY:
                pGS = shader;
                OutputDebugStringA("Succesfully open geometry shader file \n"); 
                break;
            default:
                throw std::runtime_error("Unsupported shader extension: " + shaderPath.string());
            }
        }

        // Attacher shaders
        PSOCreateInfo.pVS = pVS;
        PSOCreateInfo.pPS = pPS;
        if (pGS) PSOCreateInfo.pGS = pGS;

        AddVariablePSO(Vars, numVars, PSOCreateInfo, pPSO);

        // Stocker
        PipelineEntry entry{ pPSO, SRB };
        m_pipelines[name] = entry;

        return { &m_pipelines[name], true };
    }

    PipelineManager::PipelineEntry* PipelineManager::create2(
        const std::string& name,
        const std::vector<std::filesystem::path>& shaderPaths,
        GraphicsPipelineStateCreateInfo& PSOCreateInfo,
        Diligent::ShaderResourceVariableDesc Vars[],
        size_t numVars)
    {

        if (auto it = m_pipelines.find(name); it != m_pipelines.end())
            {
                OutputDebugStringA("[PIPELINE MANAGER] already exists \n");
                return &it->second;
            }
            

        Diligent::RefCntAutoPtr<Diligent::IPipelineState> pPSO;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> SRB;

        RefCntAutoPtr<IShader> pVS, pPS, pGS;
        for (const auto& shaderPath : shaderPaths)
        {
            Diligent::SHADER_TYPE type = GetShaderTypeFromExtension(shaderPath);
            auto shader = createShader(shaderPath);


            switch (type)
            {
            case SHADER_TYPE_VERTEX:
                pVS = shader;
                OutputDebugStringA("Succesfully open vertex shader file \n");
                break;
            case SHADER_TYPE_PIXEL:
                pPS = shader;
                OutputDebugStringA("Succesfully open pixel shader file \n");
                break;
            case SHADER_TYPE_GEOMETRY:
                pGS = shader;
                OutputDebugStringA("Succesfully open geometry shader file \n");
                break;
            default:
                throw std::runtime_error("Unsupported shader extension: " + shaderPath.string());
            }
        }

        // Attacher shaders
        PSOCreateInfo.pVS = pVS;
        PSOCreateInfo.pPS = pPS;
        if (pGS) PSOCreateInfo.pGS = pGS;

        AddVariablePSO(Vars, numVars, PSOCreateInfo, pPSO);
        std::string msg = "WARN:" + name + " not null before CreateGraphicsPipelineState\n";

        //OutputDebugStringA(msg.c_str());
        // Stocker
        PipelineEntry entry{ pPSO, SRB };
        m_pipelines[name] = entry;

        return &m_pipelines[name];
    }


    /*PipelineManager::PipelineEntry* PipelineManager::CreateGraphicsPipeline(GraphicsPipelineStateCreateInfo& PSOCreateInfo)
    {
        RefCntAutoPtr<IPipelineState> pPSO;
        m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &pPSO);

        // Créer SRB
        RefCntAutoPtr<IShaderResourceBinding> pSRB;
        pPSO->CreateShaderResourceBinding(&pSRB, true);

        // Stocker
        PipelineEntry entry{ PSOCreateInfo, pPSO, pSRB };
        m_pipelines["name"] = entry;
        return &m_pipelines["name"];
    }*/

    


    void PipelineManager::AddVariablePSO(Diligent::ShaderResourceVariableDesc Vars[], size_t numVars, GraphicsPipelineStateCreateInfo& PSOCreateInfo, Diligent::RefCntAutoPtr<Diligent::IPipelineState>& m_pPSO)
    {

        PSOCreateInfo.PSODesc.ResourceLayout.Variables = Vars;
        PSOCreateInfo.PSODesc.ResourceLayout.NumVariables = numVars;
        PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

        // Pas d’input layout car PointList
        InputLayoutDesc LayoutDesc = {};
        LayoutDesc.NumElements = 0;
        LayoutDesc.LayoutElements = nullptr;
        PSOCreateInfo.GraphicsPipeline.InputLayout = LayoutDesc;

        m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_pPSO);
        //pipelineData.device->CreateGraphicsPipelineState(PSOCreateInfo, &m_pPSO);
    }


    PipelineManager::PipelineEntry* PipelineManager::get(const std::string& name)
    {
        if (m_pipelines.find(name) == m_pipelines.end())
            throw std::runtime_error("Pipeline not found: " + name);
        return &m_pipelines[name];
    }

    RefCntAutoPtr<IShader> PipelineManager::createShader(const std::filesystem::path& path)
    {

        
        // Charger fichier source
        std::ifstream file(path.string());
        if (!file.is_open())
            
            ThrowShaderError("Failed to open shader file: " + path.string());
            //throw std::runtime_error("Failed to open shader file: " + path.string());


        ShaderCreateInfo ShaderCI;
        ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
        ShaderCI.Desc.UseCombinedTextureSamplers = true;
        ShaderCI.CompileFlags = SHADER_COMPILE_FLAG_PACK_MATRIX_ROW_MAJOR;

        RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
        m_pEngineFactory->CreateDefaultShaderSourceStreamFactory(nullptr, &pShaderSourceFactory);
        //pipelineData.engineFactory->CreateDefaultShaderSourceStreamFactory(nullptr, &pShaderSourceFactory);

        ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;


        auto shaderType = GetShaderTypeFromExtension(path);
        ShaderCI.Desc.ShaderType = shaderType;
        ShaderCI.EntryPoint = "main";
        //ShaderCI.Desc.Name = path.filename().string().c_str();
        //ShaderCI.FilePath = path.string().c_str();

        std::string fileName = path.filename().string();
        ShaderCI.Desc.Name = fileName.c_str();

        std::string filePathStr = path.string();
        ShaderCI.FilePath = filePathStr.c_str();


        //OutputDebugStringA(("Succesfully open shader file : " + path.string()).c_str());


        RefCntAutoPtr<IShader> shader;
        m_pDevice->CreateShader(ShaderCI, &shader);
        //pipelineData.device->CreateShader(ShaderCI, &shader);

        if (!shader)
            ThrowShaderError("Failed to create shader: " + path.string());

        return shader;
    }

    void PipelineManager::ThrowShaderError(const std::string& message)
    {
        // Envoi à la sortie debug Visual Studio
        OutputDebugStringA((message + "\n").c_str());

        // Lever l'exception
        throw std::runtime_error(message);
    }



} // namespace rvtx::dil

#include "rvtx/dil/utils/program.hpp"


#include <rvtx/core/logger.hpp>


#include "Common/interface/RefCntAutoPtr.hpp"
#include "Graphics/GraphicsEngine/interface/Shader.h"
#include "Graphics/GraphicsEngine/interface/PipelineState.h"
#include "Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"
#include "Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "Graphics/GraphicsEngine/interface/RenderDevice.h"

using namespace Diligent;

namespace rvtx::dil
{

    ProgramDiligent::ProgramDiligent(std::vector<std::filesystem::path> shaderPaths) : m_shaderPaths(std::move(shaderPaths)) {}


    void ProgramDiligent::create(const std::string& name)
    {
        if (!m_name.empty())
        {
            rvtx::logger::warning("Program already created");
            return;
        }

        m_name = name;
        rvtx::logger::debug("Program {} prepared for PSO creation.", m_name);
    }


    Diligent::RefCntAutoPtr<Diligent::IShader> ProgramManagerDiligent::createShader(const std::filesystem::path& path)
    {
        rvtx::logger::debug("Creating shader: {}", path.filename().string());

        const std::string name = path.string();
        const std::size_t hash = std::hash<std::string>{}(name);

        // Check if shader already exists
        auto it = m_shaders.find(hash);
        if (it != m_shaders.end())
        {
            rvtx::logger::debug("Shader already exists: {}", name);
            return it->second;
        }

        // Determine shader type from extension
        Diligent::SHADER_TYPE shaderType = Diligent::SHADER_TYPE_UNKNOWN;
        auto ext = path.extension().string();

        if (ext == ".vert") shaderType = Diligent::SHADER_TYPE_VERTEX;
        else if (ext == ".frag") shaderType = Diligent::SHADER_TYPE_PIXEL;
        else if (ext == ".geom") shaderType = Diligent::SHADER_TYPE_GEOMETRY;
        else if (ext == ".comp") shaderType = Diligent::SHADER_TYPE_COMPUTE;
        else if (ext == ".tesc") shaderType = Diligent::SHADER_TYPE_HULL;
        else if (ext == ".tese") shaderType = Diligent::SHADER_TYPE_DOMAIN;
        else
        {
            rvtx::logger::error("Invalid shader extension: {}", ext);
            return {};
        }

        // Load shader source
        std::filesystem::path fullPath = m_programRoot / path;
        const std::string src = rvtx::read(fullPath);
        if (src.empty())
        {
            rvtx::logger::error("Shader file empty or not found: {}", fullPath.string());
            return {};
        }

        // Create ShaderCreateInfo
        Diligent::ShaderCreateInfo shaderCI = {};
        shaderCI.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
        shaderCI.Desc.ShaderType = shaderType;
        shaderCI.EntryPoint = "main";
        shaderCI.FilePath = fullPath.string().c_str();
        shaderCI.Source = src.c_str();

        Diligent::RefCntAutoPtr<Diligent::IShader> shader;
        m_Device->CreateShader(shaderCI, &shader);

        if (!shader)
        {
            rvtx::logger::error("Failed to create shader: {}", name);
            return {};
        }

        rvtx::logger::debug("Shader created: {}", name);

        m_shaders[hash] = shader;
        return shader;
    }

    


    /*ProgramDiligent* ProgramManagerDiligent::create(const std::string& name, const std::vector<std::filesystem::path>& paths)
    {
        if (m_programs.find(name) != m_programs.end())
        {
            rvtx::logger::debug("Program {} already exists!", name);
            return m_programs[name].get();
        }

        std::vector<RefCntAutoPtr<IShader>> shaders;
        for (const auto& shaderPath : paths)
        {
            auto shader = createShader(shaderPath);
            if (shader)
                shaders.push_back(shader);
        }
        GraphicsPipelineStateCreateInfo PSOCreateInfo = {};
        PSOCreateInfo.PSODesc.Name = name.c_str();

        for (const auto& shader : shaders)
        {
            switch (shader->GetDesc().ShaderType)
            {
            case SHADER_TYPE_VERTEX:
                PSOCreateInfo.pVS = shader;
                break;
            case SHADER_TYPE_PIXEL:
                PSOCreateInfo.pPS = shader;
                break;
            case SHADER_TYPE_GEOMETRY:
                PSOCreateInfo.pGS = shader;
                break;
            case SHADER_TYPE_COMPUTE:
                // Compute pipeline needs a different path (not GraphicsPipeline)
                break;
            default:
                rvtx::logger::warning("Unhandled shader type in program {}", name);
                break;
            }
        }

        RefCntAutoPtr<IPipelineState> pso;
        m_Device->CreateGraphicsPipelineState(PSOCreateInfo, &pso);

        RefCntAutoPtr<IShaderResourceBinding> srb;
        pso->CreateShaderResourceBinding(&srb, true);

        auto prog = std::make_unique<ProgramDiligent>(name, pso, srb, paths);
        m_programs[name] = std::move(prog);
        return m_programs[name].get();

    }*/
}

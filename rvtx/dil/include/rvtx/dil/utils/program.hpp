#ifndef RVTX_DIL_UTILS_PROGRAM_HPP
#define RVTX_DIL_UTILS_PROGRAM_HPP

#include "rvtx/core/filesystem.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <filesystem>

#include "Common/interface/RefCntAutoPtr.hpp"
#include "Graphics/GraphicsEngine/interface/Shader.h"
#include "Graphics/GraphicsEngine/interface/PipelineState.h"
#include "Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"
#include "Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "Graphics/GraphicsEngine/interface/RenderDevice.h"

namespace rvtx::dil
{
    class ProgramDiligent
    {
    public:
        ProgramDiligent(std::vector<std::filesystem::path> shaderPaths);

        void create(const std::string& name);

        void use(Diligent::IDeviceContext* context);

        const std::vector<std::filesystem::path>& getShaderPaths() const
        {
            return m_shaderPaths;
        }

        const std::string& getName() const
        {
            return m_name;
        }

    private:
        std::string m_name;
        std::vector<std::filesystem::path> m_shaderPaths;

        Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_pso;
        Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_srb;

        friend class ProgramManagerDiligent;
    };

    class ProgramManagerDiligent
    {
    public:
        ProgramManagerDiligent(Diligent::IRenderDevice* device,
            const std::filesystem::path& programRoot = {});

        ProgramDiligent* create(const std::string& name,
            const std::vector<std::filesystem::path>& paths);

        ProgramDiligent* get(const std::string& name);

        Diligent::RefCntAutoPtr<Diligent::IShader> createShader(const std::filesystem::path& path);

    private:
        Diligent::IRenderDevice* m_Device = nullptr;
        std::filesystem::path m_programRoot;

        std::unordered_map<std::string, std::unique_ptr<ProgramDiligent>> m_programs;
        std::unordered_map<std::size_t, Diligent::RefCntAutoPtr<Diligent::IShader>> m_shaders;
    };

} // namespace rvtx::dil



#endif // RVTX_DIL_UTILS_PROGRAM_HPP

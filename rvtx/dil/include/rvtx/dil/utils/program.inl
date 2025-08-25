// program.inl


#include "Common/interface/RefCntAutoPtr.hpp"
#include "Graphics/GraphicsEngine/interface/Shader.h"
#include "Graphics/GraphicsEngine/interface/PipelineState.h"
#include "Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"


namespace rvtx::dil
{
    inline Diligent::SHADER_TYPE toDiligentShaderType(ShaderType type)
    {
        using dg = Diligent;

        switch (type)
        {
        case ShaderType::Vertex:
            return dg::SHADER_TYPE_VERTEX;
        case ShaderType::Fragment:
            return dg::SHADER_TYPE_PIXEL;
        case ShaderType::Geometry:
            return dg::SHADER_TYPE_GEOMETRY;
        case ShaderType::Compute:
            return dg::SHADER_TYPE_COMPUTE;
        case ShaderType::TessellationEvaluation:
            return dg::SHADER_TYPE_DOMAIN;
        case ShaderType::TessellationControl:
            return dg::SHADER_TYPE_HULL;
        default:
            return dg::SHADER_TYPE_UNKNOWN;
        }
    }
}

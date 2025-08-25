#ifndef RVTX_SYSTEM_MATERIAL_PARAMETERS_HPP
#define RVTX_SYSTEM_MATERIAL_PARAMETERS_HPP

#include <glm/vec3.hpp>

namespace rvtx
{
    struct MaterialParameters
    {
        glm::vec3 baseColor = { 0.f, 0.f, 0.f };
        float     roughness = 0.1f;

        glm::vec3 emissive  = { 0.f, 0.f, 0.f };
        float     metallic = 0.f;

        glm::vec3 transmittance = { 0.f, 0.f, 0.f };
        float     atDistance    = 0.f;

        float ior                  = 0.f;
        float specularTransmission = 0.f;
        float specularTint         = 0.f;
        float clearcoat            = 0.f;

        float clearcoatGloss = 0.f;
    };
} // namespace rvtx
#endif // RVTX_SYSTEM_MATERIAL_PARAMETERS_HPP

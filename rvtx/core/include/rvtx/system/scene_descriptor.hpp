#ifndef RVTX_SYSTEM_SCENE_DESCRIPTOR_HPP
#define RVTX_SYSTEM_SCENE_DESCRIPTOR_HPP

#include <filesystem>
#include <vector>

#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/material_parameters.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx
{
    struct SceneDescriptor;

    SceneDescriptor parse( const std::filesystem::path & file );
    void            save( const std::filesystem::path & outFile, const SceneDescriptor & scene );

    struct EntityDescriptor
    {
        enum Type
        {
            Molecule,
            Mesh,
        };

        Type                  type;
        std::filesystem::path path;
        Transform             transform;

        // For molecule
        struct Representation
        {
            RepresentationType type;
            MaterialParameters materialParameters;
            ColorMode          colorMode;
            float              probeRadius;
        };
        std::vector<Representation> representations {};

        // For mesh
        MaterialParameters materialParameters;
    };

    struct RendererDescriptor
    {
        // General
        bool       useWindowViewport;
        glm::uvec2 viewport;
        uint32_t   width;
        uint32_t   height;

        // For ray tracing
        uint32_t raysPerPixel;
        uint32_t maxRayBounces;
        uint32_t samplesPerPixel;
    };

    struct CameraDescriptor
    {
        Transform                transform;
        rvtx::Camera::Projection projectionType;
        float                    fov;
        glm::vec3                depthOfField;

        Camera::Target target;

        // Only used on load
        uint32_t targetEntity;
    };

    struct BackgroundDescriptor
    {
        bool                  useExr;
        std::filesystem::path exrPath;
        glm::vec3             color;
        float                 weight;
    };

    struct WindowDescriptor
    {
        glm::uvec2  viewport;
        uint32_t    width;
        uint32_t    height;
        std::string title;
        bool        shown;
    };

    struct SceneDescriptor
    {
        std::string          name;
        WindowDescriptor     windowDescriptor;
        RendererDescriptor   rendererDescriptor;
        BackgroundDescriptor backgroundDescriptor;
        CameraDescriptor     cameraDescriptor;

        std::vector<EntityDescriptor> entities;
    };
} // namespace rvtx

#endif // RVTX_SYSTEM_SCENE_DESCRIPTOR_HPP

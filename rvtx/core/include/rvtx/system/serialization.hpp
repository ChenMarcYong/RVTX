#ifndef RVTX_SYSTEM_SERIALIZATION_HPP
#define RVTX_SYSTEM_SERIALIZATION_HPP

#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <nlohmann/json.hpp>

#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/scene_descriptor.hpp"
#include "rvtx/system/transform.hpp"

NLOHMANN_JSON_NAMESPACE_BEGIN

// glm
template<>
struct adl_serializer<glm::uvec2>
{
    static void to_json( json & j, const glm::uvec2 & opt ) { j = { opt.x, opt.y }; }

    static void from_json( const json & j, glm::uvec2 & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.x = j[ 0 ].get<uint32_t>();
        opt.y = j[ 1 ].get<uint32_t>();
    }
};
template<>
struct adl_serializer<glm::vec3>
{
    static void to_json( json & j, const glm::vec3 & opt ) { j = { opt.x, opt.y, opt.z }; }

    static void from_json( const json & j, glm::vec3 & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.x = j[ 0 ].get<float>();
        opt.y = j[ 1 ].get<float>();
        opt.z = j[ 2 ].get<float>();
    }
};
template<>
struct adl_serializer<glm::vec4>
{
    static void to_json( json & j, const glm::vec4 & opt ) { j = { opt.x, opt.y, opt.z, opt.w }; }

    static void from_json( const json & j, glm::vec4 & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.x = j[ 0 ].get<float>();
        opt.y = j[ 1 ].get<float>();
        opt.z = j[ 2 ].get<float>();
        opt.w = j[ 3 ].get<float>();
    }
};
template<>
struct adl_serializer<glm::quat>
{
    static void to_json( json & j, const glm::quat & opt ) { j = { opt.w, opt.x, opt.y, opt.z }; }

    static void from_json( const json & j, glm::quat & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.w = j[ 0 ].get<float>();
        opt.x = j[ 1 ].get<float>();
        opt.y = j[ 2 ].get<float>();
        opt.z = j[ 3 ].get<float>();
    }
};

// vtx
template<>
struct adl_serializer<rvtx::MaterialParameters>
{
    static void to_json( json & j, const rvtx::MaterialParameters & opt )
    {
        j[ "base_color" ] = opt.baseColor;

        j[ "emissive" ] = opt.emissive;

        j[ "transmittance" ][ "color" ]       = opt.transmittance;
        j[ "transmittance" ][ "at_distance" ] = opt.atDistance;

        j[ "ior" ] = opt.ior;

        j[ "specular" ][ "roughness" ]    = opt.roughness;
        j[ "specular" ][ "metalness" ]    = opt.metallic;
        j[ "specular" ][ "transmission" ] = opt.specularTransmission;
        j[ "specular" ][ "tint" ]         = opt.specularTint;

        j[ "clearcoat" ][ "weight" ] = opt.clearcoat;
        j[ "clearcoat" ][ "gloss" ]  = opt.clearcoatGloss;
    }

    static void from_json( const json & j, rvtx::MaterialParameters & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.baseColor = j[ "base_color" ].get<glm::vec3>();

        opt.emissive = j[ "emissive" ].get<glm::vec3>();

        opt.transmittance = j[ "transmittance" ][ "color" ].get<glm::vec3>();
        opt.atDistance    = j[ "transmittance" ][ "at_distance" ].get<float>();

        opt.ior = j[ "ior" ].get<float>();

        opt.roughness            = j[ "specular" ][ "roughness" ].get<float>();
        opt.metallic             = j[ "specular" ][ "metallic" ].get<float>();
        opt.specularTransmission = j[ "specular" ][ "transmission" ].get<float>();
        opt.specularTint         = j[ "specular" ][ "tint" ].get<float>();

        opt.clearcoat      = j[ "clearcoat" ][ "weight" ].get<float>();
        opt.clearcoatGloss = j[ "clearcoat" ][ "gloss" ].get<float>();
    }
};

// pt
template<>
struct adl_serializer<rvtx::Transform>
{
    static void to_json( json & j, const rvtx::Transform & opt )
    {
        j[ "position" ] = opt.position;
        j[ "rotation" ] = opt.rotation;
    }

    static void from_json( const json & j, rvtx::Transform & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.position = j[ "position" ].get<glm::vec3>();
        opt.rotation = j[ "rotation" ].get<glm::quat>();
    }
};

template<>
struct adl_serializer<rvtx::EntityDescriptor::Representation>
{
    static void to_json( json & j, const rvtx::EntityDescriptor::Representation & opt )
    {
        j[ "representation_type" ] = opt.type;
        j[ "material" ]            = opt.materialParameters;
        j[ "color_mode" ]          = opt.colorMode;
        j[ "probe_radius" ]        = opt.probeRadius;
    }

    static void from_json( const json & j, rvtx::EntityDescriptor::Representation & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.type               = j[ "representation_type" ].get<rvtx::RepresentationType>();
        opt.materialParameters = j[ "material" ].get<rvtx::MaterialParameters>();
        opt.colorMode          = j[ "color_mode" ].get<rvtx::ColorMode>();
        opt.probeRadius        = j[ "probe_radius" ].get<float>();
    }
};

template<>
struct adl_serializer<rvtx::EntityDescriptor>
{
    static void to_json( json & j, const rvtx::EntityDescriptor & opt )
    {
        j[ "entity_type" ] = opt.type;
        j[ "path" ]        = opt.path;
        j[ "transform" ]   = opt.transform;

        switch ( opt.type )
        {
        case rvtx::EntityDescriptor::Molecule: j[ "representations" ] = opt.representations; break;
        case rvtx::EntityDescriptor::Mesh: j[ "material" ] = opt.materialParameters; break;
        }
    }

    static void from_json( const json & j, rvtx::EntityDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.type      = j[ "entity_type" ].get<rvtx::EntityDescriptor::Type>();
        opt.path      = j[ "path" ].get<std::string>();
        opt.transform = j[ "transform" ].get<rvtx::Transform>();

        switch ( opt.type )
        {
        case rvtx::EntityDescriptor::Molecule:
        {
            opt.representations = j[ "representations" ].get<std::vector<rvtx::EntityDescriptor::Representation>>();
            break;
        }
        case rvtx::EntityDescriptor::Mesh:
        {
            opt.materialParameters = j[ "material" ].get<rvtx::MaterialParameters>();
            break;
        }
        }
    }
};

template<>
struct adl_serializer<rvtx::WindowDescriptor>
{
    static void to_json( json & j, const rvtx::WindowDescriptor & opt )
    {
        j[ "title" ]  = opt.title;
        j[ "width" ]  = opt.width;
        j[ "height" ] = opt.height;
        j[ "shown" ]  = opt.shown;
    }

    static void from_json( const json & j, rvtx::WindowDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.title      = j[ "title" ].get<std::string>();
        opt.width      = j[ "width" ].get<std::uint32_t>();
        opt.height     = j[ "height" ].get<uint32_t>();
        opt.viewport = { opt.width, opt.height };
        opt.shown      = j[ "shown" ].get<bool>();
    }
};

template<>
struct adl_serializer<rvtx::RendererDescriptor>
{
    static void to_json( json & j, const rvtx::RendererDescriptor & opt )
    {
        j[ "use_window_viewport" ] = opt.useWindowViewport;
        j[ "viewport_width" ]      = opt.viewport.x;
        j[ "viewport_height" ]     = opt.viewport.y;

        j[ "ray_tracing_settings" ][ "max_ray_bounces" ]   = opt.maxRayBounces;
        j[ "ray_tracing_settings" ][ "rays_per_pixel" ]    = opt.raysPerPixel;
        j[ "ray_tracing_settings" ][ "samples_per_pixel" ] = opt.samplesPerPixel;
    }

    static void from_json( const json & j, rvtx::RendererDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.useWindowViewport = j[ "use_window_viewport" ].get<bool>();

        opt.width  = j[ "viewport_width" ].get<uint32_t>();
        opt.height = j[ "viewport_height" ].get<uint32_t>();

        opt.viewport = { opt.width, opt.height };

        opt.maxRayBounces   = j[ "ray_tracing_settings" ][ "max_ray_bounces" ].get<uint32_t>();
        opt.raysPerPixel    = j[ "ray_tracing_settings" ][ "rays_per_pixel" ].get<uint32_t>();
        opt.samplesPerPixel = j[ "ray_tracing_settings" ][ "samples_per_pixel" ].get<uint32_t>();
    }
};

template<>
struct adl_serializer<rvtx::BackgroundDescriptor>
{
    static void to_json( json & j, const rvtx::BackgroundDescriptor & opt )
    {
        j[ "exr_path" ] = opt.useExr ? opt.exrPath : "";
        j[ "color" ]    = opt.color;
        j[ "weight" ]   = opt.weight;
    }

    static void from_json( const json & j, rvtx::BackgroundDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.exrPath = j[ "exr_path" ].get<std::string>();
        opt.useExr  = !opt.exrPath.empty();
        opt.color   = j[ "color" ].get<glm::vec3>();
        opt.weight  = j[ "weight" ].get<float>();
    }
};

template<>
struct adl_serializer<rvtx::Camera::Target>
{
    static void to_json( json & j, const rvtx::Camera::Target & opt )
    {
        j[ "position" ] = opt.position;
        j[ "distance" ] = opt.distance;
    }

    static void from_json( const json & j, rvtx::Camera::Target & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.position = j[ "position" ].get<glm::vec3>();
        opt.distance = j[ "distance" ].get<float>();
    }
};

template<>
struct adl_serializer<rvtx::CameraDescriptor>
{
    static void to_json( json & j, const rvtx::CameraDescriptor & opt )
    {
        j[ "transform" ]       = opt.transform;
        j[ "projection_type" ] = opt.projectionType;
        j[ "fov" ]             = opt.fov;
        j[ "depth_of_field" ]  = opt.depthOfField;
        j[ "target" ]          = opt.target;
    }

    static void from_json( const json & j, rvtx::CameraDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.transform      = j[ "transform" ].get<rvtx::Transform>();
        opt.projectionType = j[ "projection_type" ].get<rvtx::Camera::Projection>();
        opt.fov            = j[ "fov" ].get<float>();
        opt.depthOfField   = j[ "depth_of_field" ].get<glm::vec3>();
        opt.target         = j[ "target" ].get<rvtx::Camera::Target>();
        opt.targetEntity   = j[ "target_entity" ].get<int32_t>();
    }
};

template<>
struct adl_serializer<rvtx::SceneDescriptor>
{
    static void to_json( json & j, const rvtx::SceneDescriptor & opt )
    {
        j[ "name" ]                  = opt.name;
        j[ "window_descriptor" ]     = opt.windowDescriptor;
        j[ "renderer_descriptor" ]   = opt.rendererDescriptor;
        j[ "background_descriptor" ] = opt.backgroundDescriptor;
        j[ "camera_descriptor" ]     = opt.cameraDescriptor;
        j[ "entities" ]              = opt.entities;
    }

    static void from_json( const json & j, rvtx::SceneDescriptor & opt )
    {
        if ( j.is_null() )
        {
            opt = {};
            return;
        }

        opt.name                 = j[ "name" ].get<std::string>();
        opt.windowDescriptor     = j[ "window_descriptor" ].get<rvtx::WindowDescriptor>();
        opt.rendererDescriptor   = j[ "renderer_descriptor" ].get<rvtx::RendererDescriptor>();
        opt.backgroundDescriptor = j[ "background_descriptor" ].get<rvtx::BackgroundDescriptor>();
        opt.cameraDescriptor     = j[ "camera_descriptor" ].get<rvtx::CameraDescriptor>();
        opt.entities             = j[ "entities" ].get<std::vector<rvtx::EntityDescriptor>>();
    }
};
NLOHMANN_JSON_NAMESPACE_END

#endif // RVTX_SYSTEM_SERIALIZATION_HPP

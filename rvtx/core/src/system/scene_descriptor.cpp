#include "rvtx/system/scene_descriptor.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "rvtx/core/filesystem.hpp"
#include "rvtx/core/logger.hpp"
#include "rvtx/system/serialization.hpp"

namespace rvtx
{
    SceneDescriptor parse( const std::filesystem::path & path )
    {
        if ( !std::filesystem::exists( path ) )
        {
            logger::error( "Scene file '{}' not found.", path.string() );
            throw std::runtime_error( fmt::format( "Scene file '{}' not found.", path.string() ) );
        }

        std::ifstream file { path };

        if ( !file )
        {
            logger::error( "Error while loading scene file '{}'.", path.string() );
            throw std::runtime_error( fmt::format( "Error while loading scene file '{}'.", path.string() ) );
        }

        nlohmann::json json = nlohmann::json::parse( file );
        return json.get<SceneDescriptor>();
    }

    void save( const std::filesystem::path & outFile, const SceneDescriptor & scene )
    {
        nlohmann::json json {};
        json = scene;

        rvtx::save( outFile, json.dump( 4 ) );
    }
} // namespace rvtx

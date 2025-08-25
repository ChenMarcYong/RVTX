#include "rvtx/core/image.hpp"

#include <fstream>

#include <tinyexr.h>

#include "rvtx/core/filesystem.hpp"
#include "rvtx/core/logger.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace rvtx
{
    std::vector<float> loadExr( std::filesystem::path path, uint32_t & width, uint32_t & height, uint32_t & channels )
    {
        const std::string pathStr = path.string();
        if ( !std::filesystem::exists( path ) )
        {
            rvtx::logger::error( "Can't find file at {}.", pathStr );
            return {};
        }

        float *      out       = nullptr; // width * height * RGBA
        const char * layerName = nullptr;
        const char * err       = nullptr;
        int          wwidth = 0, hheight = 0;

        std::vector<float> result = {};
        const int          ret    = LoadEXRWithLayer( &out, &wwidth, &hheight, pathStr.c_str(), layerName, &err );
        if ( ret != TINYEXR_SUCCESS )
        {
            if ( !err )
            {
                rvtx::logger::error( "{}", err );
                FreeEXRErrorMessage( err ); // release memory of error message.
            }
        }
        else
        {
            width    = static_cast<uint32_t>( wwidth );
            height   = static_cast<uint32_t>( hheight );
            channels = 4;

            result.resize( width * height * channels );
            std::memcpy( result.data(), out, width * height * channels * sizeof( float ) );
            free( out ); // release memory of image data
        }

        return result;
    }

    void save( std::filesystem::path path,
               ConstSpan<uint32_t>   data,
               uint32_t              width,
               uint32_t              height,
               std::string_view      text,
               bool                  flip )
    {
        stbi_flip_vertically_on_write( flip );
        stbi_write_png_compression_level = 0;

        const std::string str = path.string();
        if ( !stbi_write_png( str.c_str(), width, height, 4, data.ptr, 0 ) )
            rvtx::logger::error( "Failed to save snapshot at {}", str );

        if ( text.empty() )
            return;

        rvtx::save( path.replace_extension( ".txt" ), text );
    }
} // namespace rvtx
#include "rvtx/gl/utils/snapshot.hpp"

#include <vector>

#include <GL/gl3w.h>

#include <stb_image_write.h>

#include "rvtx/core/filesystem.hpp"
#include "rvtx/core/logger.hpp"

namespace rvtx::gl
{
    void snapshot( const std::filesystem::path & outPath,
                   GLuint                        framebuffer,
                   uint32_t                      width,
                   uint32_t                      height,
                   bool                          logInfos )
    {
        auto image = std::vector<uint8_t>( width * height * 4 );
        glBindFramebuffer( GL_FRAMEBUFFER, framebuffer );
        glReadnPixels(
            0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, static_cast<GLsizei>( image.size() ), image.data() );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        stbi_flip_vertically_on_write( true );
        stbi_write_png_compression_level = 0;

        const std::string str = outPath.string();
        if ( !stbi_write_png( str.c_str(), width, height, 4, image.data(), 0 ) )
            logger::error( "Failed to save snapshot at {}", str );

        if ( logInfos )
            logger::info( "Snapshot saved at '{}'.", str );
    }

    void snapshot( const std::filesystem::path & path,
                   ConstSpan<uint32_t>           data,
                   uint32_t                      width,
                   uint32_t                      height,
                   bool                          flip,
                   bool                          logInfos )
    {
        stbi_flip_vertically_on_write( flip );
        stbi_write_png_compression_level = 0;

        const std::string str = path.string();
        if ( !stbi_write_png( str.c_str(), width, height, 4, data.ptr, 0 ) )
            logger::error( "Failed to save snapshot at {}", str );

        if ( logInfos )
            logger::info( "Snapshot saved at '{}'.", str );
    }
} // namespace rvtx::gl

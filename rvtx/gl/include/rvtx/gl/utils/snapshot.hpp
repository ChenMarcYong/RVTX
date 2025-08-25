#ifndef RVTX_GL_UTILS_SNAPSHOT_HPP
#define RVTX_GL_UTILS_SNAPSHOT_HPP

#include <filesystem>

#include "rvtx/core/type.hpp"
#include "rvtx/gl/core/fwd.hpp"

namespace rvtx::gl
{
    void snapshot( const std::filesystem::path & outPath,
                   GLuint                        framebuffer,
                   uint32_t                      width,
                   uint32_t                      height,
                   bool                          logInfos = true );

    void snapshot( const std::filesystem::path & path,
                   ConstSpan<uint32_t>           data,
                   uint32_t                      width,
                   uint32_t                      height,
                   bool                          flip     = false,
                   bool                          logInfos = true );
} // namespace rvtx::gl

#endif // RVTX_GL_UTILS_SNAPSHOT_HPP

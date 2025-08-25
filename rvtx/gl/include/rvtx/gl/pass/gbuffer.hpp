#ifndef RVTX_GL_PASS_GBUFFER_HPP
#define RVTX_GL_PASS_GBUFFER_HPP

#include <functional>

#include "rvtx/gl/core/fwd.hpp"

namespace rvtx::gl
{
    class ProgramManager;
    using GeometryDraw = std::function<void()>;
    class GBufferPass
    {
      public:
        GBufferPass() = default;
        GBufferPass( uint32_t width, uint32_t height );

        GBufferPass( const GBufferPass & )             = delete;
        GBufferPass & operator=( const GBufferPass & ) = delete;

        GBufferPass( GBufferPass && ) noexcept;
        GBufferPass & operator=( GBufferPass && ) noexcept;

        ~GBufferPass();

        GLuint getGeometryTexture() const;
        GLuint getMaterialTexture() const;
        GLuint getIdsTexture() const;
        GLuint getDepthTexture() const;

        void resize( uint32_t width, uint32_t height );
        void render( GeometryDraw geometryDraw = {} );

      private:
        GLuint m_fbo                                   = GL_INVALID_VALUE;
        GLuint m_viewPositionsNormalsCompressedTexture = GL_INVALID_VALUE;
        GLuint m_colorsTexture                         = GL_INVALID_VALUE;
        GLuint m_idsTexture                            = GL_INVALID_VALUE;
        GLuint m_depthTexture                          = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_PASS_GBUFFER_HPP

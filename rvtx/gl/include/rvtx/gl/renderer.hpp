#ifndef RVTX_GL_RENDERER_HPP
#define RVTX_GL_RENDERER_HPP

#include <cstdint>

#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/pass/gbuffer.hpp"
#include "rvtx/gl/pass/post_process.hpp"
#include "rvtx/gl/window.hpp"
#include "rvtx/system/scene.hpp"

namespace rvtx::gl
{
    class Renderer
    {
      public:
        Renderer() = default;
        Renderer( Window & window, ProgramManager & programManager );
        Renderer( Window & window, ProgramManager & programManager, const glm::uvec2 viewport );

        Renderer( const Renderer & )             = delete;
        Renderer & operator=( const Renderer & ) = delete;

        Renderer( Renderer && );
        Renderer & operator=( Renderer && );

        ~Renderer();

        inline void              enableUI( bool enable );
        inline void              setGeometry( std::unique_ptr<GeometryHandler> && geometry );
        inline GLuint            getRendererTextureHandle() const;
        inline GLuint            getFramebuffer() const;
        inline PostProcessPass & getPostProcess();

        void resize( uint32_t width, uint32_t height );
        void render(
            const Camera &              camera,
            const Scene &               scene,
            const std::function<void()> updateUIFunction = [] {} );
        void render( const std::function<void()> & updateUIFunction = [] {} );

      private:
        uint32_t m_width;
        uint32_t m_height;

        Window * m_window;

        bool m_enableUI = true;

        GBufferPass     m_gBufferPass;
        PostProcessPass m_postProcessPass;

        std::unique_ptr<GeometryHandler> m_geometry {};

        GLuint m_rendererTextureFBO = GL_INVALID_VALUE;
        GLuint m_rendererTexture    = GL_INVALID_VALUE;

        GLuint m_currentFBO = GL_INVALID_VALUE;

        void resizeRendererTexture();
    };
} // namespace rvtx::gl

#include "rvtx/gl/renderer.inl"

#endif // RVTX_GL_RENDERER_HPP

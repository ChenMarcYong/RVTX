#include "rvtx/gl/renderer.hpp"

namespace rvtx::gl
{
    inline void Renderer::enableUI( bool enable ) { m_enableUI = enable; }

    inline void Renderer::setGeometry( std::unique_ptr<GeometryHandler> && geometry )
    {
        m_geometry = std::move( geometry );
    }

    inline GLuint Renderer::getRendererTextureHandle() const { return m_rendererTexture; }

    inline GLuint Renderer::getFramebuffer() const { return m_currentFBO; }

    inline PostProcessPass & Renderer::getPostProcess() { return m_postProcessPass; }
} // namespace rvtx::gl

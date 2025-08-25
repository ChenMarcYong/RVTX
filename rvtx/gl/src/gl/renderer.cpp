#include "rvtx/gl/renderer.hpp"

#include <GL/gl3w.h>
#include <SDL.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl2.h>

#include "rvtx/core/logger.hpp"
#include "rvtx/gl/window.hpp"

namespace rvtx::gl
{
    Renderer::Renderer( Window & window, ProgramManager & programManager ) :
        m_width( window.getWidth() ), m_height( window.getHeight() ), m_window( &window )
    {
        SDL_GL_SetSwapInterval( 0 );

        m_gBufferPass     = GBufferPass( m_width, m_height );
        m_postProcessPass = PostProcessPass( m_width, m_height, programManager );

        m_postProcessPass.setGeometricTexture( m_gBufferPass.getGeometryTexture() );
        m_postProcessPass.setMaterialTexture( m_gBufferPass.getMaterialTexture() );
        m_postProcessPass.setDepthTexture( m_gBufferPass.getDepthTexture() );

        resizeRendererTexture();
    }

    Renderer::Renderer( Window & window, ProgramManager & programManager, const glm::uvec2 viewport ) :
        m_width( viewport.x ), m_height( viewport.y ), m_window( &window )
    {
        SDL_GL_SetSwapInterval( 0 );

        m_gBufferPass     = GBufferPass( m_width, m_height );
        m_postProcessPass = PostProcessPass( m_width, m_height, programManager );

        m_postProcessPass.setGeometricTexture( m_gBufferPass.getGeometryTexture() );
        m_postProcessPass.setMaterialTexture( m_gBufferPass.getMaterialTexture() );
        m_postProcessPass.setDepthTexture( m_gBufferPass.getDepthTexture() );

        resizeRendererTexture();
    }

    Renderer::Renderer( Renderer && other )
    {
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_window, other.m_window );
        std::swap( m_gBufferPass, other.m_gBufferPass );
        std::swap( m_postProcessPass, other.m_postProcessPass );
        std::swap( m_geometry, other.m_geometry );
    }

    Renderer & Renderer::operator=( Renderer && other )
    {
        std::swap( m_width, other.m_width );
        std::swap( m_height, other.m_height );
        std::swap( m_window, other.m_window );
        std::swap( m_gBufferPass, other.m_gBufferPass );
        std::swap( m_postProcessPass, other.m_postProcessPass );
        std::swap( m_geometry, other.m_geometry );

        return *this;
    }

    Renderer::~Renderer()
    {
        if ( m_rendererTextureFBO != GL_INVALID_VALUE )
            glDeleteFramebuffers( 1, &m_rendererTextureFBO );
        if ( m_rendererTexture != GL_INVALID_VALUE )
            glDeleteTextures( 1, &m_rendererTexture );
    };

    void Renderer::resize( uint32_t width, uint32_t height )
    {
        if ( m_width == width && m_height == height )
            return;

        m_width  = width;
        m_height = height;

        m_gBufferPass.resize( m_width, m_height );
        m_postProcessPass.resize( m_width, m_height );

        resizeRendererTexture();
    }

    void Renderer::render( const Camera & camera, const Scene & scene, const std::function<void()> updateUIFunction )
    {
        glViewport( 0, 0, m_width, m_height );
        m_gBufferPass.render(
            [ this, &scene, &camera ]
            {
                if ( m_geometry )
                    m_geometry->render( camera, scene );
            } );
        m_postProcessPass.render( camera );

        SDL_Window * handle = m_window->getHandle();

        if ( m_enableUI )
        {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame( handle );

            ImGui::NewFrame();
            updateUIFunction();
            ImGui::Render();
        }

        SDL_GL_MakeCurrent( handle, m_window->getContext() );
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glBindFramebuffer( GL_READ_FRAMEBUFFER, m_postProcessPass.getFramebuffer() );
        glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 );
        glBlitFramebuffer( 0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST );

        if ( m_enableUI )
            ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

        SDL_GL_SwapWindow( handle );

        m_currentFBO = m_postProcessPass.getFramebuffer();
    }

    void Renderer::render( const std::function<void()> & updateUIFunction )
    {
        glViewport( 0, 0, m_width, m_height );

        SDL_Window * handle = m_window->getHandle();

        if ( m_enableUI )
        {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame( handle );

            ImGui::NewFrame();
            updateUIFunction();
            ImGui::Render();
        }

        SDL_GL_MakeCurrent( handle, m_window->getContext() );

        // Copy framebuffer to default framebuffer
        // glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glBindFramebuffer( GL_READ_FRAMEBUFFER, m_rendererTextureFBO );
        glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 );
        glBlitFramebuffer( 0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST );

        if ( m_enableUI )
            ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

        SDL_GL_SwapWindow( handle );

        m_currentFBO = m_rendererTextureFBO;
    }

    void Renderer::resizeRendererTexture()
    {
        if ( m_rendererTextureFBO != GL_INVALID_VALUE )
            glDeleteFramebuffers( 1, &m_rendererTextureFBO );
        if ( m_rendererTexture != GL_INVALID_VALUE )
            glDeleteTextures( 1, &m_rendererTexture );

        glGenFramebuffers( 1, &m_rendererTextureFBO );
        glBindFramebuffer( GL_FRAMEBUFFER, m_rendererTextureFBO );

        glGenTextures( 1, &m_rendererTexture );
        glBindTexture( GL_TEXTURE_2D, m_rendererTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rendererTexture, 0 );
        glBindTexture( GL_TEXTURE_2D, 0 );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        const GLenum fboStatus = glCheckFramebufferStatus( GL_FRAMEBUFFER );
        if ( fboStatus != GL_FRAMEBUFFER_COMPLETE )
            logger::debug( "Framebuffer not complete: {}", fboStatus );

        const GLenum glstatus = glGetError();
        if ( glstatus != GL_NO_ERROR )
            logger::debug( "Error in GL call: {}", glstatus );
    }
} // namespace rvtx::gl

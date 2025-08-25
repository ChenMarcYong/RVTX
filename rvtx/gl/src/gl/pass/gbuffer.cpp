#include "rvtx/gl/pass/gbuffer.hpp"

#include <array>

#include <GL/gl3w.h>

#include "rvtx/core/logger.hpp"

namespace rvtx::gl
{
    GBufferPass::GBufferPass( uint32_t width, uint32_t height )
    {
        glGenFramebuffers( 1, &m_fbo );

        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glGenTextures( 1, &m_viewPositionsNormalsCompressedTexture );
        glBindTexture( GL_TEXTURE_2D, m_viewPositionsNormalsCompressedTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr );

        glGenTextures( 1, &m_colorsTexture );
        glBindTexture( GL_TEXTURE_2D, m_colorsTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr );

        glGenTextures( 1, &m_idsTexture );
        glBindTexture( GL_TEXTURE_2D, m_idsTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr );

        glGenTextures( 1, &m_depthTexture );
        glBindTexture( GL_TEXTURE_2D, m_depthTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr );

        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_viewPositionsNormalsCompressedTexture, 0 );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_colorsTexture, 0 );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_idsTexture, 0 );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTexture, 0 );

        constexpr std::array<GLenum, 3> drawBuffers
            = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };

        glDrawBuffers( 3, drawBuffers.data() );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        const GLenum fboStatus = glCheckFramebufferStatus( GL_FRAMEBUFFER );
        if ( fboStatus != GL_FRAMEBUFFER_COMPLETE )
            rvtx::logger::error( "Framebuffer not complete: {}", fboStatus );

        const GLenum glstatus = glGetError();
        if ( glstatus != GL_NO_ERROR )
            rvtx::logger::error( "Error in GL call: {}", glstatus );
    }

    GBufferPass::GBufferPass( GBufferPass && other ) noexcept
    {
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_viewPositionsNormalsCompressedTexture, other.m_viewPositionsNormalsCompressedTexture );
        std::swap( m_colorsTexture, other.m_colorsTexture );
        std::swap( m_idsTexture, other.m_idsTexture );
        std::swap( m_depthTexture, other.m_depthTexture );
    }

    GBufferPass & GBufferPass::operator=( GBufferPass && other ) noexcept
    {
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_viewPositionsNormalsCompressedTexture, other.m_viewPositionsNormalsCompressedTexture );
        std::swap( m_colorsTexture, other.m_colorsTexture );
        std::swap( m_idsTexture, other.m_idsTexture );
        std::swap( m_depthTexture, other.m_depthTexture );

        return *this;
    }

    GBufferPass::~GBufferPass()
    {
        if ( glIsFramebuffer( m_fbo ) )
            glDeleteFramebuffers( 1, &m_fbo );

        if ( glIsTexture( m_viewPositionsNormalsCompressedTexture ) )
            glDeleteTextures( 1, &m_viewPositionsNormalsCompressedTexture );

        if ( glIsTexture( m_colorsTexture ) )
            glDeleteTextures( 1, &m_colorsTexture );

        if ( glIsTexture( m_idsTexture ) )
            glDeleteTextures( 1, &m_idsTexture );

        if ( glIsTexture( m_depthTexture ) )
            glDeleteTextures( 1, &m_depthTexture );
    }

    GLuint GBufferPass::getGeometryTexture() const { return m_viewPositionsNormalsCompressedTexture; }
    GLuint GBufferPass::getMaterialTexture() const { return m_colorsTexture; }
    GLuint GBufferPass::getIdsTexture() const { return m_idsTexture; }
    GLuint GBufferPass::getDepthTexture() const { return m_depthTexture; }

    void GBufferPass::resize( uint32_t width, uint32_t height )
    {
        glBindTexture( GL_TEXTURE_2D, m_viewPositionsNormalsCompressedTexture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr );

        glBindTexture( GL_TEXTURE_2D, m_colorsTexture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr );

        glBindTexture( GL_TEXTURE_2D, m_idsTexture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr );

        glBindTexture( GL_TEXTURE_2D, m_depthTexture );
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr );
    }

    void GBufferPass::render( GeometryDraw geometryDraw )
    {
        glEnable( GL_DEPTH_TEST );

        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        if ( geometryDraw )
            geometryDraw();

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
        glDisable( GL_DEPTH_TEST );
    }

} // namespace rvtx::gl

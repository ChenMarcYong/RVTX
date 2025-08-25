#include "rvtx/gl/pass/post_process.hpp"

#include <GL/gl3w.h>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <rvtx/core/math.hpp>
#include <rvtx/system/camera.hpp>

#include "rvtx/gl/core/sampling.hpp"
#include "rvtx/gl/utils/program.hpp"

namespace rvtx::gl
{
    LinearizeDepthPostProcess::LinearizeDepthPostProcess( uint32_t width, uint32_t height, ProgramManager & manager ) :
        Pass( width, height )
    {
        glCreateVertexArrays( 1, &m_vao );

        glGenFramebuffers( 1, &m_fbo );
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glGenTextures( 1, &m_drawTexture );
        glBindTexture( GL_TEXTURE_2D, m_drawTexture );
        // TODO: LINEAR useful ?
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_drawTexture, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        m_program = manager.create( "LinearizeDepth", { "geometry/full_screen.vert", "shading/linearize_depth.frag" } );
        m_uClipInfoLoc   = glGetUniformLocation( m_program->getId(), "uClipInfo" );
        m_uIsPerspective = glGetUniformLocation( m_program->getId(), "uIsPerspective" );
    }

    LinearizeDepthPostProcess::LinearizeDepthPostProcess( LinearizeDepthPostProcess && other ) noexcept :
        Pass( std::move( other ) )
    {
        std::swap( m_program, other.m_program );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_drawTexture, other.m_drawTexture );
        std::swap( m_inputTexture, other.m_inputTexture );
        std::swap( m_uClipInfoLoc, other.m_uClipInfoLoc );
        std::swap( m_uIsPerspective, other.m_uIsPerspective );
    }

    LinearizeDepthPostProcess & LinearizeDepthPostProcess::operator=( LinearizeDepthPostProcess && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_drawTexture, other.m_drawTexture );
        std::swap( m_inputTexture, other.m_inputTexture );
        std::swap( m_uClipInfoLoc, other.m_uClipInfoLoc );
        std::swap( m_uIsPerspective, other.m_uIsPerspective );

        Pass::operator=( std::move( other ) );

        return *this;
    }

    LinearizeDepthPostProcess::~LinearizeDepthPostProcess()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
        if ( glIsFramebuffer( m_fbo ) )
            glDeleteFramebuffers( 1, &m_fbo );
        if ( glIsTexture( m_drawTexture ) )
            glDeleteTextures( 1, &m_drawTexture );
    }

    void   LinearizeDepthPostProcess::setInputTexture( GLuint inputTexture ) { m_inputTexture = inputTexture; }
    GLuint LinearizeDepthPostProcess::getTexture() const { return m_drawTexture; }

    void LinearizeDepthPostProcess::resize( uint32_t width, uint32_t height )
    {
        glBindTexture( GL_TEXTURE_2D, m_drawTexture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );
    }

    void LinearizeDepthPostProcess::render( const Camera & camera )
    {
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, m_inputTexture );

        m_program->use();
        // // TODO don't update each frame
        const float camNear = camera.zNear;
        const float camFar  = camera.zFar;
        glUniform4f( m_uClipInfoLoc, camNear * camFar, camFar, camFar - camNear, camNear );
        glUniform1i( m_uIsPerspective, camera.isPerspective() );

        glBindVertexArray( m_vao );
        glDrawArrays( GL_TRIANGLE_STRIP, 0, 3 );
        glBindVertexArray( 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    }

    SSAOPostProcess::SSAOPostProcess( uint32_t width, uint32_t height, ProgramManager & manager ) :
        Pass( width, height )
    {
        glCreateVertexArrays( 1, &m_vao );

        glGenFramebuffers( 1, &m_fbo );
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glGenTextures( 1, &m_texture );
        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        m_program = manager.create( "SSAO", { "geometry/full_screen.vert", "shading/ssao.frag" } );

        m_uProjMatrixLoc  = glGetUniformLocation( m_program->getId(), "uProjMatrix" );
        m_uAoKernelLoc    = glGetUniformLocation( m_program->getId(), "uAoKernel" );
        m_uAoIntensityLoc = glGetUniformLocation( m_program->getId(), "uAoIntensity" );
        m_uKernelSizeLoc  = glGetUniformLocation( m_program->getId(), "uKernelSize" );
        m_uNoiseSizeLoc   = glGetUniformLocation( m_program->getId(), "uNoiseSize" );

        // generate random ao kernel
        m_aoKernel = std::vector<glm::vec3>( m_kernelSize );

        for ( uint32_t i = 0; i < m_kernelSize; i++ )
        {
            // sample on unit hemisphere
            glm::vec3 v = cosineWeightedHemisphere( random( 0.f, 1.f ), random( 0.f, 1.f ) );

            // scale sample within the hemisphere
            v *= random( 0.f, 1.f );

            // accelerating interpolation (distance from center reduces when number of points grow up)
            float scale = static_cast<float>( i ) / static_cast<float>( m_kernelSize );
            scale       = rvtx::lerp( 0.01f, 1.f, scale * scale );
            v *= scale;
            m_aoKernel[ i ] = v;
        }

        // generate noise texture
        std::vector<glm::vec3> noise {};
        noise.resize( m_noiseTextureSize * m_noiseTextureSize );

        for ( std::size_t i = 0; i < noise.size(); ++i )
        {
            noise[ i ] = glm::vec3( glm::linearRand( -1.f, 1.f ), glm::linearRand( -1.f, 1.f ), 0.f );
            noise[ i ] = glm::normalize( noise[ i ] );
        }
        glGenTextures( 1, &m_noiseTexture );
        glBindTexture( GL_TEXTURE_2D, m_noiseTexture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        // repeat tile over the image
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB16F, m_noiseTextureSize, m_noiseTextureSize, 0, GL_RGB, GL_FLOAT, noise.data() );

        m_program->use();
        glUniform3fv( m_uAoKernelLoc, m_kernelSize, reinterpret_cast<const GLfloat *>( m_aoKernel.data() ) );
        glUniform1i( m_uAoIntensityLoc, 2.f /* RVTX_SETTING().aoIntensity  */ );
        glUniform1i( m_uKernelSizeLoc, m_kernelSize );
        glUniform1f( m_uNoiseSizeLoc, static_cast<float>( m_noiseTextureSize ) );
    }

    SSAOPostProcess::SSAOPostProcess( SSAOPostProcess && other ) noexcept : Pass( std::move( other ) )
    {
        std::swap( m_program, other.m_program );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_texture, other.m_texture );
        std::swap( m_noiseTexture, other.m_noiseTexture );
        std::swap( m_depthTexture, other.m_depthTexture );
        std::swap( m_geometricTexture, other.m_geometricTexture );
        std::swap( m_uProjMatrixLoc, other.m_uProjMatrixLoc );
        std::swap( m_uAoKernelLoc, other.m_uAoKernelLoc );
        std::swap( m_uKernelSizeLoc, other.m_uKernelSizeLoc );
        std::swap( m_uAoIntensityLoc, other.m_uAoIntensityLoc );
        std::swap( m_uNoiseSizeLoc, other.m_uNoiseSizeLoc );
        std::swap( m_kernelSize, other.m_kernelSize );
        std::swap( m_noiseTextureSize, other.m_noiseTextureSize );
        std::swap( m_aoKernel, other.m_aoKernel );
    }

    SSAOPostProcess & SSAOPostProcess::operator=( SSAOPostProcess && other ) noexcept
    {
        std::swap( m_program, other.m_program );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_texture, other.m_texture );
        std::swap( m_noiseTexture, other.m_noiseTexture );
        std::swap( m_depthTexture, other.m_depthTexture );
        std::swap( m_geometricTexture, other.m_geometricTexture );
        std::swap( m_uProjMatrixLoc, other.m_uProjMatrixLoc );
        std::swap( m_uAoKernelLoc, other.m_uAoKernelLoc );
        std::swap( m_uKernelSizeLoc, other.m_uKernelSizeLoc );
        std::swap( m_uAoIntensityLoc, other.m_uAoIntensityLoc );
        std::swap( m_uNoiseSizeLoc, other.m_uNoiseSizeLoc );
        std::swap( m_kernelSize, other.m_kernelSize );
        std::swap( m_noiseTextureSize, other.m_noiseTextureSize );
        std::swap( m_aoKernel, other.m_aoKernel );

        Pass::operator=( std::move( other ) );

        return *this;
    }

    SSAOPostProcess::~SSAOPostProcess()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
        if ( glIsFramebuffer( m_fbo ) )
            glDeleteFramebuffers( 1, &m_fbo );
        if ( glIsTexture( m_texture ) )
            glDeleteTextures( 1, &m_texture );
        if ( glIsTexture( m_noiseTexture ) )
            glDeleteTextures( 1, &m_noiseTexture );
    }

    void SSAOPostProcess::setDepthTexture( GLuint depthTexture ) { m_depthTexture = depthTexture; }

    void SSAOPostProcess::setGeometricTexture( GLuint geometricTexture ) { m_geometricTexture = geometricTexture; }

    GLuint SSAOPostProcess::getTexture() const { return m_texture; }

    void SSAOPostProcess::resize( uint32_t width, uint32_t height )
    {
        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_FLOAT, nullptr );
    }

    void SSAOPostProcess::render( const Camera & camera )
    {
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, m_geometricTexture );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, m_noiseTexture );
        glActiveTexture( GL_TEXTURE2 );
        glBindTexture( GL_TEXTURE_2D, m_depthTexture );

        m_program->use();
        //
        // // TODO don't update each frame
        glUniform3fv( m_uAoKernelLoc, m_kernelSize, reinterpret_cast<const GLfloat *>( m_aoKernel.data() ) );
        glUniform1i( m_uAoIntensityLoc, 5.f /*  RVTX_SETTING().aoIntensity  */ );

        glUniformMatrix4fv( m_uProjMatrixLoc, 1, GL_FALSE, glm::value_ptr( camera.getProjectionMatrix() ) );
        glUniform1i( m_uKernelSizeLoc, m_kernelSize );
        glUniform1f( m_uNoiseSizeLoc, static_cast<float>( m_noiseTextureSize ) );

        glBindVertexArray( m_vao );
        glDrawArrays( GL_TRIANGLE_STRIP, 0, 3 );
        glBindVertexArray( 0 );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, 0 );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, 0 );
        glActiveTexture( GL_TEXTURE2 );
        glBindTexture( GL_TEXTURE_2D, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    }

    BlurPostProcess::BlurPostProcess( uint32_t width, uint32_t height, ProgramManager & manager ) :
        Pass( width, height )
    {
        glCreateVertexArrays( 1, &m_vao );

        // first pass fbo/texture
        glGenFramebuffers( 1, &m_fboFirstPass );
        glBindFramebuffer( GL_FRAMEBUFFER, m_fboFirstPass );

        glGenTextures( 1, &m_textureFirstPass );
        glBindTexture( GL_TEXTURE_2D, m_textureFirstPass );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_textureFirstPass, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        glGenFramebuffers( 1, &m_fbo );
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glGenTextures( 1, &m_texture );
        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        m_program = manager.create( "Blur", { "geometry/full_screen.vert", "shading/bilateral_blur.frag" } );

        m_uBlurSizeLoc            = glGetUniformLocation( m_program->getId(), "uBlurSize" );
        m_uInvDirectionTexSizeLoc = glGetUniformLocation( m_program->getId(), "uInvDirectionTexSize" );

        m_program->use();
        glUniform1i( m_uBlurSizeLoc, 14 /* RVTX_SETTING().aoBlurSize */ );

        const float value = 1.f;
        glClearTexImage( m_texture, 0, GL_RED, GL_FLOAT, &value );
    }

    BlurPostProcess::BlurPostProcess( BlurPostProcess && other ) noexcept : Pass( std::move( other ) )
    {
        std::swap( m_vao, other.m_vao );
        std::swap( m_inputTexture1, other.m_inputTexture1 );
        std::swap( m_inputTexture2, other.m_inputTexture2 );
        std::swap( m_program, other.m_program );
        std::swap( m_fboFirstPass, other.m_fboFirstPass );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_textureFirstPass, other.m_textureFirstPass );
        std::swap( m_texture, other.m_texture );
        std::swap( m_uBlurSizeLoc, other.m_uBlurSizeLoc );
        std::swap( m_uInvDirectionTexSizeLoc, other.m_uInvDirectionTexSizeLoc );
    }

    BlurPostProcess & BlurPostProcess::operator=( BlurPostProcess && other ) noexcept
    {
        std::swap( m_vao, other.m_vao );
        std::swap( m_inputTexture1, other.m_inputTexture1 );
        std::swap( m_inputTexture2, other.m_inputTexture2 );
        std::swap( m_program, other.m_program );
        std::swap( m_fboFirstPass, other.m_fboFirstPass );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_textureFirstPass, other.m_textureFirstPass );
        std::swap( m_texture, other.m_texture );
        std::swap( m_uBlurSizeLoc, other.m_uBlurSizeLoc );
        std::swap( m_uInvDirectionTexSizeLoc, other.m_uInvDirectionTexSizeLoc );

        Pass::operator=( std::move( other ) );

        return *this;
    }

    BlurPostProcess::~BlurPostProcess()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
        if ( glIsFramebuffer( m_fboFirstPass ) )
            glDeleteFramebuffers( 1, &m_fboFirstPass );
        if ( glIsTexture( m_fboFirstPass ) )
            glDeleteTextures( 1, &m_textureFirstPass );
        if ( glIsFramebuffer( m_fbo ) )
            glDeleteFramebuffers( 1, &m_fbo );
        if ( glIsTexture( m_texture ) )
            glDeleteTextures( 1, &m_texture );
    }

    void BlurPostProcess::setInputTexture1( GLuint texture ) { m_inputTexture1 = texture; }
    void BlurPostProcess::setInputTexture2( GLuint texture ) { m_inputTexture2 = texture; }

    GLuint BlurPostProcess::getTexture() const { return m_texture; }

    void BlurPostProcess::resize( uint32_t width, uint32_t height )
    {
        glBindTexture( GL_TEXTURE_2D, m_textureFirstPass );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr );

        const float value = 1.f;
        glClearTexImage( m_texture, 0, GL_RED, GL_FLOAT, &value );
    }

    void BlurPostProcess::render()
    {
        // TODO: clean up !!!!!!!!!!!!!!!
        glBindFramebuffer( GL_FRAMEBUFFER, m_fboFirstPass );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, m_inputTexture1 );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, m_inputTexture2 );

        m_program->use();
        // TODO don't update each frame
        glUniform1i( m_uBlurSizeLoc, 14 /* RVTX_SETTING().aoBlurSize  */ );
        glUniform2i( m_uInvDirectionTexSizeLoc, 1, 0 );

        glBindVertexArray( m_vao );
        glDrawArrays( GL_TRIANGLE_STRIP, 0, 3 );
        glBindVertexArray( 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, 0 );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, 0 );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, m_textureFirstPass );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, m_inputTexture2 );

        glUniform2i( m_uInvDirectionTexSizeLoc, 0, 1 );

        glBindVertexArray( m_vao );
        glDrawArrays( GL_TRIANGLE_STRIP, 0, 3 );
        glBindVertexArray( 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    }

    ShadingPostProcess::ShadingPostProcess( uint32_t width, uint32_t height, ProgramManager & manager ) :
        Pass( width, height )
    {
        glCreateVertexArrays( 1, &m_vao );

        glGenFramebuffers( 1, &m_fbo );
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glGenTextures( 1, &m_texture );
        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr );

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture, 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        m_toonShading    = manager.create( "ToonShading", { "geometry/full_screen.vert", "shading/toon.frag" } );
        m_diffuseShading = manager.create( "DiffuseShading", { "geometry/full_screen.vert", "shading/diffuse.frag" } );
        m_glossyShading  = manager.create( "GlossyShading", { "geometry/full_screen.vert", "shading/glossy.frag" } );
        m_flatShading    = manager.create( "FlatShading", { "geometry/full_screen.vert", "shading/flat.frag" } );

        // Use setting value.
        set();
    }

    ShadingPostProcess::ShadingPostProcess( ShadingPostProcess && other ) noexcept : Pass( std::move( other ) )
    {
        std::swap( m_toonShading, other.m_toonShading );
        std::swap( m_diffuseShading, other.m_diffuseShading );
        std::swap( m_glossyShading, other.m_glossyShading );
        std::swap( m_flatShading, other.m_flatShading );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_texture, other.m_texture );
        std::swap( m_geometricTexture, other.m_geometricTexture );
        std::swap( m_materialTexture, other.m_materialTexture );
        std::swap( m_occlusionTexture, other.m_occlusionTexture );
        std::swap( m_uBackgroundColorLoc, other.m_uBackgroundColorLoc );
        std::swap( m_uFogNear, other.m_uFogNear );
        std::swap( m_uFogFar, other.m_uFogFar );
        std::swap( m_uFogDensity, other.m_uFogDensity );
        std::swap( m_uFogColor, other.m_uFogColor );
        std::swap( m_uLightPosition, other.m_uLightPosition );
        std::swap( m_uLightColor, other.m_uLightColor );
    }

    ShadingPostProcess & ShadingPostProcess::operator=( ShadingPostProcess && other ) noexcept
    {
        std::swap( m_toonShading, other.m_toonShading );
        std::swap( m_diffuseShading, other.m_diffuseShading );
        std::swap( m_glossyShading, other.m_glossyShading );
        std::swap( m_flatShading, other.m_flatShading );
        std::swap( m_vao, other.m_vao );
        std::swap( m_fbo, other.m_fbo );
        std::swap( m_texture, other.m_texture );
        std::swap( m_geometricTexture, other.m_geometricTexture );
        std::swap( m_materialTexture, other.m_materialTexture );
        std::swap( m_occlusionTexture, other.m_occlusionTexture );
        std::swap( m_uBackgroundColorLoc, other.m_uBackgroundColorLoc );
        std::swap( m_uFogNear, other.m_uFogNear );
        std::swap( m_uFogFar, other.m_uFogFar );
        std::swap( m_uFogDensity, other.m_uFogDensity );
        std::swap( m_uFogColor, other.m_uFogColor );
        std::swap( m_uLightPosition, other.m_uLightPosition );
        std::swap( m_uLightColor, other.m_uLightColor );

        Pass::operator=( std::move( other ) );

        return *this;
    }

    ShadingPostProcess::~ShadingPostProcess()
    {
        if ( glIsVertexArray( m_vao ) )
            glDeleteVertexArrays( 1, &m_vao );
        if ( glIsFramebuffer( m_fbo ) )
            glDeleteFramebuffers( 1, &m_fbo );
        if ( glIsTexture( m_texture ) )
            glDeleteTextures( 1, &m_texture );
    }

    void ShadingPostProcess::setGeometricTexture( GLuint geometricTexture ) { m_geometricTexture = geometricTexture; }

    void ShadingPostProcess::setMaterialTexture( GLuint materialTexture ) { m_materialTexture = materialTexture; }
    void ShadingPostProcess::setOcclusionTexture( GLuint occlusionTexture ) { m_occlusionTexture = occlusionTexture; }

    GLuint ShadingPostProcess::getTexture() const { return m_texture; }
    GLuint ShadingPostProcess::getFramebuffer() const { return m_fbo; }

    void ShadingPostProcess::resize( uint32_t width, uint32_t height )
    {
        glBindTexture( GL_TEXTURE_2D, m_texture );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr );
    }

    void ShadingPostProcess::render( const Camera & camera )
    {
        glBindFramebuffer( GL_FRAMEBUFFER, m_fbo );

        glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, m_geometricTexture );
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_2D, m_materialTexture );

        glActiveTexture( GL_TEXTURE2 );
        // If SSAO/Blur disabled, texture is previoulsy cleared.
        glBindTexture( GL_TEXTURE_2D, m_occlusionTexture );

        m_flatShading->use();
        //
        // // TODO: do not update each frame
        // const Color::Rgb & bgColor = RVTX_SETTING().backgroundColor;
        glm::vec3 bgColor = glm::vec3( 1.f, 1.f, 1.f );
        glUniform3f( m_uBackgroundColorLoc, bgColor.r, bgColor.g, bgColor.b );
        // // TODO: no need for flat shading
        // // TODO: let the user choose where's the light
        // // TODO: distinguish "view" and "world" lights
        const glm::vec4 & lightPosition = camera.getViewMatrix() * glm::vec4( camera.transform->position, 1.f );
        glUniform3f( m_uLightPosition, lightPosition.x, lightPosition.y, lightPosition.z );
        // const Color::Rgb & lightColor = RVTX_SETTING().lightColor;
        const glm::vec3 lightColor = glm::vec3( 1.f );
        glUniform3f( m_uLightColor, lightColor.r, lightColor.g, lightColor.b );

        glBindVertexArray( m_vao );
        glDrawArrays( GL_TRIANGLE_STRIP, 0, 3 );
        glBindVertexArray( 0 );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    }

    void ShadingPostProcess::set()
    {
        // switch ( RVTX_SETTING().shading )
        // {
        // case SHADING::TOON: _currentShading = _toonShading; break;
        // case SHADING::GLOSSY: _currentShading = _glossyShading; break;
        // case SHADING::FLAT_COLOR: _currentShading = _flatShading; break;
        // case SHADING::DIFFUSE:
        // default: _currentShading = _diffuseShading;
        // }

        m_currentShading = m_flatShading;

        m_currentShading->use();
        m_uBackgroundColorLoc = glGetUniformLocation( m_currentShading->getId(), "uBackgroundColor" );
        m_uFogNear            = glGetUniformLocation( m_currentShading->getId(), "uFogNear" );
        m_uFogFar             = glGetUniformLocation( m_currentShading->getId(), "uFogFar" );
        m_uFogDensity         = glGetUniformLocation( m_currentShading->getId(), "uFogDensity" );
        m_uFogColor           = glGetUniformLocation( m_currentShading->getId(), "uFogColor" );
        m_uLightPosition      = glGetUniformLocation( m_currentShading->getId(), "uLightPosition" );
        m_uLightColor         = glGetUniformLocation( m_currentShading->getId(), "uLightColor" );
    }

    PostProcessPass::PostProcessPass( uint32_t width, uint32_t height, ProgramManager & manager ) :
        Pass( width, height ), m_linearizeDepth( width, height, manager ), m_ssao( width, height, manager ),
        m_blur( width, height, manager ), m_shading( width, height, manager )
    {
        m_ssao.setDepthTexture( m_linearizeDepth.getTexture() );
        m_blur.setInputTexture1( m_ssao.getTexture() );
        m_blur.setInputTexture2( m_linearizeDepth.getTexture() );
        m_shading.setOcclusionTexture( m_blur.getTexture() );
    }

    PostProcessPass::PostProcessPass( PostProcessPass && other ) noexcept : Pass( std::move( other ) )
    {
        std::swap( m_linearizeDepth, other.m_linearizeDepth );
        std::swap( m_ssao, other.m_ssao );
        std::swap( m_blur, other.m_blur );
        std::swap( m_shading, other.m_shading );
    }

    PostProcessPass & PostProcessPass::operator=( PostProcessPass && other ) noexcept
    {
        std::swap( m_linearizeDepth, other.m_linearizeDepth );
        std::swap( m_ssao, other.m_ssao );
        std::swap( m_blur, other.m_blur );
        std::swap( m_shading, other.m_shading );

        Pass::operator=( std::move( other ) );

        return *this;
    }

    PostProcessPass::~PostProcessPass() = default;

    void PostProcessPass::setGeometricTexture( GLuint geometricTexture )
    {
        m_shading.setGeometricTexture( geometricTexture );
        m_ssao.setGeometricTexture( geometricTexture );
    }

    void PostProcessPass::setDepthTexture( GLuint depthTexture ) { m_linearizeDepth.setInputTexture( depthTexture ); }

    void PostProcessPass::setMaterialTexture( GLuint materialTexture )
    {
        m_shading.setMaterialTexture( materialTexture );
    }

    GLuint PostProcessPass::getFramebuffer() const { return m_shading.getFramebuffer(); }

    void PostProcessPass::resize( uint32_t width, uint32_t height )
    {
        m_linearizeDepth.resize( width, height );
        m_ssao.resize( width, height );
        m_blur.resize( width, height );
        m_shading.resize( width, height );
    }

    void PostProcessPass::render( const Camera & camera )
    {
        m_linearizeDepth.render( camera );

        if ( m_enableAO )
        {
            m_ssao.render( camera );
            m_blur.render();
        }

        m_shading.render( camera );
    }
} // namespace rvtx::gl

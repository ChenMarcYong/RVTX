#ifndef RVTX_GL_PASS_POSTPROCESS_HPP
#define RVTX_GL_PASS_POSTPROCESS_HPP

#include <vector>

#include <glm/vec3.hpp>

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/pass/pass.hpp"

namespace rvtx
{
    struct Camera;
}

namespace rvtx::gl
{
    class Program;
    class ProgramManager;

    class LinearizeDepthPostProcess : public Pass
    {
      public:
        LinearizeDepthPostProcess() = default;
        LinearizeDepthPostProcess( uint32_t width, uint32_t height, ProgramManager & manager );

        LinearizeDepthPostProcess( const LinearizeDepthPostProcess & )             = delete;
        LinearizeDepthPostProcess & operator=( const LinearizeDepthPostProcess & ) = delete;

        LinearizeDepthPostProcess( LinearizeDepthPostProcess && ) noexcept;
        LinearizeDepthPostProcess & operator=( LinearizeDepthPostProcess && ) noexcept;

        ~LinearizeDepthPostProcess() override;

        void   setInputTexture( GLuint inputTexture );
        GLuint getTexture() const;

        void resize( uint32_t width, uint32_t height ) override;
        void render( const Camera & camera );

      private:
        Program * m_program = nullptr;

        GLuint m_vao            = GL_INVALID_VALUE;
        GLuint m_fbo            = GL_INVALID_VALUE;
        GLuint m_drawTexture    = GL_INVALID_VALUE;
        GLuint m_inputTexture   = GL_INVALID_VALUE;
        GLint  m_uClipInfoLoc   = GL_INVALID_INDEX;
        GLint  m_uIsPerspective = GL_INVALID_INDEX;
    };

    class SSAOPostProcess : public Pass
    {
      public:
        SSAOPostProcess() = default;
        SSAOPostProcess( uint32_t width, uint32_t height, ProgramManager & manager );

        SSAOPostProcess( const SSAOPostProcess & )             = delete;
        SSAOPostProcess & operator=( const SSAOPostProcess & ) = delete;

        SSAOPostProcess( SSAOPostProcess && ) noexcept;
        SSAOPostProcess & operator=( SSAOPostProcess && ) noexcept;

        ~SSAOPostProcess() override;

        void   setDepthTexture( GLuint depthTexture );
        void   setGeometricTexture( GLuint compressedSpatialTexture );
        GLuint getTexture() const;

        void resize( uint32_t width, uint32_t height ) override;
        void render( const Camera & camera );

      private:
        Program * m_program = nullptr;

        GLuint m_vao = GL_INVALID_VALUE;

        GLuint m_fbo              = GL_INVALID_VALUE;
        GLuint m_texture          = GL_INVALID_VALUE;
        GLuint m_noiseTexture     = GL_INVALID_VALUE;
        GLuint m_depthTexture     = GL_INVALID_VALUE;
        GLuint m_geometricTexture = GL_INVALID_VALUE;

        GLint m_uProjMatrixLoc  = GL_INVALID_INDEX;
        GLint m_uAoKernelLoc    = GL_INVALID_INDEX;
        GLint m_uKernelSizeLoc  = GL_INVALID_INDEX;
        GLint m_uAoIntensityLoc = GL_INVALID_INDEX;
        GLint m_uNoiseSizeLoc   = GL_INVALID_INDEX;

        uint32_t               m_kernelSize       = 16;
        uint32_t               m_noiseTextureSize = 64;
        std::vector<glm::vec3> m_aoKernel;
    };

    class BlurPostProcess : public Pass
    {
      public:
        BlurPostProcess() = default;
        BlurPostProcess( uint32_t width, uint32_t height, ProgramManager & manager );

        BlurPostProcess( const BlurPostProcess & )             = delete;
        BlurPostProcess & operator=( const BlurPostProcess & ) = delete;

        BlurPostProcess( BlurPostProcess && ) noexcept;
        BlurPostProcess & operator=( BlurPostProcess && ) noexcept;

        ~BlurPostProcess() override;

        void   setInputTexture1( GLuint texture );
        void   setInputTexture2( GLuint texture );
        GLuint getTexture() const;

        void resize( uint32_t width, uint32_t height ) override;
        void render();

      private:
        Program * m_program = nullptr;

        GLuint m_vao                     = GL_INVALID_VALUE;
        GLuint m_inputTexture1           = GL_INVALID_VALUE;
        GLuint m_inputTexture2           = GL_INVALID_VALUE;
        GLuint m_fboFirstPass            = GL_INVALID_VALUE;
        GLuint m_fbo                     = GL_INVALID_VALUE;
        GLuint m_textureFirstPass        = GL_INVALID_VALUE;
        GLuint m_texture                 = GL_INVALID_VALUE;
        GLint  m_uBlurSizeLoc            = GL_INVALID_INDEX;
        GLint  m_uInvDirectionTexSizeLoc = GL_INVALID_INDEX;
    };

    class ShadingPostProcess : public Pass
    {
      public:
        ShadingPostProcess() = default;
        ShadingPostProcess( uint32_t width, uint32_t height, ProgramManager & manager );

        ShadingPostProcess( const ShadingPostProcess & )             = delete;
        ShadingPostProcess & operator=( const ShadingPostProcess & ) = delete;

        ShadingPostProcess( ShadingPostProcess && ) noexcept;
        ShadingPostProcess & operator=( ShadingPostProcess && ) noexcept;

        ~ShadingPostProcess() override;

        void   setGeometricTexture( GLuint geometricTexture );
        void   setMaterialTexture( GLuint materialTexture );
        void   setOcclusionTexture( GLuint occlusionTexture );
        GLuint getTexture() const;
        GLuint getFramebuffer() const;

        void resize( uint32_t width, uint32_t height ) override;
        void render( const Camera & camera );

      private:
        void set();

        Program * m_diffuseShading = nullptr;
        Program * m_glossyShading  = nullptr;
        Program * m_toonShading    = nullptr;
        Program * m_flatShading    = nullptr;
        Program * m_currentShading = nullptr;

        GLuint m_vao     = GL_INVALID_VALUE;
        GLuint m_fbo     = GL_INVALID_VALUE;
        GLuint m_texture = GL_INVALID_VALUE;

        GLuint m_geometricTexture = GL_INVALID_VALUE;
        GLuint m_materialTexture  = GL_INVALID_VALUE;
        GLuint m_occlusionTexture = GL_INVALID_VALUE;

        GLint m_uBackgroundColorLoc = GL_INVALID_INDEX;
        // Fog.
        GLint m_uFogNear    = GL_INVALID_INDEX;
        GLint m_uFogFar     = GL_INVALID_INDEX;
        GLint m_uFogDensity = GL_INVALID_INDEX;
        GLint m_uFogColor   = GL_INVALID_INDEX;
        // Lights.
        GLint m_uLightPosition = GL_INVALID_INDEX;
        GLint m_uLightColor    = GL_INVALID_INDEX;

        // GLSL::Program * _currentShading = nullptr;
    };

    class PostProcessPass : public Pass
    {
      public:
        PostProcessPass() = default;
        PostProcessPass( uint32_t width, uint32_t height, ProgramManager & manager );

        PostProcessPass( const PostProcessPass & )             = delete;
        PostProcessPass & operator=( const PostProcessPass & ) = delete;

        PostProcessPass( PostProcessPass && ) noexcept;
        PostProcessPass & operator=( PostProcessPass && ) noexcept;

        ~PostProcessPass() override;

        void setGeometricTexture( GLuint geometricTexture );
        void setDepthTexture( GLuint depthTexture );
        void setMaterialTexture( GLuint materialTexture );

        inline bool getEnableAO() const;
        inline void setEnableAO( bool enable );
        inline void toggleEnableAO();

        inline GLuint getLinearizedDepthTexture() const;
        inline GLuint getSSAOTexture() const;
        inline GLuint getBlurTexture() const;
        inline GLuint getShadingTexture() const;

        GLuint getFramebuffer() const;

        void resize( uint32_t width, uint32_t height ) override;
        void render( const Camera & camera );

      private:
        LinearizeDepthPostProcess m_linearizeDepth;
        SSAOPostProcess           m_ssao;
        BlurPostProcess           m_blur;
        ShadingPostProcess        m_shading;

        bool m_enableAO = true;
    };
} // namespace rvtx::gl

#include "rvtx/gl/pass/post_process.inl"

#endif // RVTX_GL_PASS_POSTPROCESS_HPP

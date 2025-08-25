#ifndef PYRVTX_PY_ENGINE_HPP
#define PYRVTX_PY_ENGINE_HPP

#include <glm/vec4.hpp>

#include "rvtx/gl/pass/gbuffer.hpp"
#include "rvtx/gl/pass/post_process.hpp"
#include "rvtx/gl/utils/program.hpp"
#include "rvtx/ux/camera_controller.hpp"

namespace rvtx
{
    namespace gl
    {
        struct GeometryHandler;
        class Window;
    } // namespace gl
} // namespace rvtx

namespace rvtx
{
    class PyEngine;
    struct PyCamera;
    struct PyScene;

    struct PyEngineView
    {
        static constexpr auto in_place_delete = true;

        PyEngine * pyengine;
    };

    class PyEngine
    {
      public:
         PyEngine( gl::Window * window, const bool enableControls, const std::string & shaderRootPath = "shaders" );
         PyEngine( const uint32_t      width,
                   const uint32_t      height,
                   const bool          enableControls,
                   const std::string & shaderRootPath = "shaders" );
        ~PyEngine();

        void render( PyScene & scene, const PyCamera * camera = nullptr );
        bool update( PyScene & scene, PyCamera * camera = nullptr );

        std::vector<unsigned char> screenshot( PyScene & scene, const PyCamera * camera = nullptr );

        void                  resize( const uint32_t width, const uint32_t height );
        GLuint                getFramebuffer() const;
        gl::PostProcessPass & getPostProcess();

        bool controlsEnabled = true;

        inline uint32_t getWidth() const { return m_width; };
        inline uint32_t getHeight() const { return m_height; };

        std::vector<uint32_t>  getIDsImage() const;
        std::vector<float>     getDepthImage() const;
        std::vector<glm::vec4> getShadingImage() const;
        std::vector<glm::vec4> getMaterialImage() const;

        gl::GBufferPass     gBufferPass;
        gl::PostProcessPass postProcessPass;

      private:
        uint32_t m_width;
        uint32_t m_height;

        gl::Window * m_window;

        std::unique_ptr<gl::GeometryHandler> m_geometryHandler {};

        GLuint m_rendererTextureFBO = GL_INVALID_VALUE;
        GLuint m_rendererTexture    = GL_INVALID_VALUE;

        GLuint m_currentFBO = GL_INVALID_VALUE;

        gl::ProgramManager programManager;

        void resizeRendererTexture();
    };
} // namespace rvtx

#endif
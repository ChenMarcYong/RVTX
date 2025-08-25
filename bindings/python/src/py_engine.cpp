#include "pyrvtx/py_engine.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <GL/gl3w.h>
#include <SDL.h>
#include <rvtx/gl/geometry/ball_and_stick_geometry.hpp>
#include <rvtx/gl/geometry/debug_primitives_geometry.hpp>
#include <rvtx/gl/geometry/sesdf_geometry.hpp>
#include <rvtx/gl/geometry/sphere_geometry.hpp>
#include <rvtx/gl/window.hpp>
#include <stb_image_write.h>

#include "pyrvtx/py_camera.hpp"
#include "pyrvtx/py_scene.hpp"
#include "rvtx/gl/geometry/Sas_geometry.hpp"
#include "rvtx/gl/geometry/mesh_geometry.hpp"
#include "rvtx/gl/geometry/sticks_geometry.hpp"

namespace rvtx
{
    PyEngine::PyEngine( gl::Window * window, const bool enableControls, const std::string & shaderRootPath ) :
        m_width( window->getWidth() ), m_height( window->getHeight() ), m_window( window ),
        controlsEnabled( enableControls )
    {
        programManager = { shaderRootPath };

        SDL_GL_SetSwapInterval( 0 );

        gBufferPass     = gl::GBufferPass( m_width, m_height );
        postProcessPass = gl::PostProcessPass( m_width, m_height, programManager );

        postProcessPass.setGeometricTexture( gBufferPass.getGeometryTexture() );
        postProcessPass.setMaterialTexture( gBufferPass.getMaterialTexture() );
        postProcessPass.setDepthTexture( gBufferPass.getDepthTexture() );

        resizeRendererTexture();

        // Creates handled geometries of the GL renderer
        auto geometryForwarder = std::make_unique<gl::GeometryForwarder>();
        geometryForwarder->add<gl::SphereHandler>( programManager );
        geometryForwarder->add<gl::SesdfHandler>( programManager );
        geometryForwarder->add<gl::BallAndStickHandler>( programManager );
        geometryForwarder->add<gl::DebugPrimitivesHandler>( programManager );
        geometryForwarder->add<gl::MeshHandler>( programManager );
        geometryForwarder->add<gl::SasHandler>( programManager );
        geometryForwarder->add<gl::SticksHandler>( programManager );
        m_geometryHandler = std::move( geometryForwarder );
    }

    PyEngine::PyEngine( const uint32_t      width,
                        const uint32_t      height,
                        const bool          enableControls,
                        const std::string & shaderRootPath ) :
        m_width( width ), m_height( height ), m_window( nullptr ), controlsEnabled( enableControls )
    {
        programManager = { shaderRootPath };

        gBufferPass     = gl::GBufferPass( m_width, m_height );
        postProcessPass = gl::PostProcessPass( m_width, m_height, programManager );

        postProcessPass.setGeometricTexture( gBufferPass.getGeometryTexture() );
        postProcessPass.setMaterialTexture( gBufferPass.getMaterialTexture() );
        postProcessPass.setDepthTexture( gBufferPass.getDepthTexture() );

        resizeRendererTexture();

        // Creates handled geometries of the GL renderer
        auto geometryForwarder = std::make_unique<gl::GeometryForwarder>();
        geometryForwarder->add<gl::SphereHandler>( programManager );
        geometryForwarder->add<gl::SesdfHandler>( programManager );
        geometryForwarder->add<gl::BallAndStickHandler>( programManager );
        geometryForwarder->add<gl::DebugPrimitivesHandler>( programManager );
        geometryForwarder->add<gl::MeshHandler>( programManager );
        m_geometryHandler = std::move( geometryForwarder );
    }

    PyEngine::~PyEngine()
    {
        if ( m_rendererTextureFBO != GL_INVALID_VALUE )
            glDeleteFramebuffers( 1, &m_rendererTextureFBO );
        if ( m_rendererTexture != GL_INVALID_VALUE )
            glDeleteTextures( 1, &m_rendererTexture );
    }

    void PyEngine::render( PyScene & scene, const PyCamera * camera )
    {
        const PyCamera * cam = camera != nullptr ? camera : scene.mainCamera;

        if ( cam == nullptr )
            throw std::runtime_error( "No camera provided in engine render call and scene main camera is null!" );

        if ( !scene.hasPyEngine() )
        {
            auto   pyEngineEntity = scene.createEntity();
            auto & pyEngineView   = pyEngineEntity.emplace<PyEngineView>();
            pyEngineView.pyengine = this;
        }

        auto pointCloudView = scene.registry.view<PyPointCloud>();
        for ( auto id : pointCloudView )
        {
            entt::handle entity     = { scene.registry, id };
            auto &       primitives = entity.get<PyPointCloud>();

            if ( primitives.holder == nullptr )
                primitives.holder = &entity.emplace<gl::DebugPrimitivesHolder>();

            if ( primitives.needsUpdate )
            {
                primitives.update();
                primitives.needsUpdate = false;
            }
        }

        auto graphView = scene.registry.view<PyGraph>();
        for ( auto id : graphView )
        {
            entt::handle entity     = { scene.registry, id };
            auto &       primitives = entity.get<PyGraph>();

            if ( primitives.holder == nullptr )
                primitives.holder = &entity.emplace<gl::DebugPrimitivesHolder>();

            if ( primitives.needsUpdate )
            {
                primitives.update();
                primitives.needsUpdate = false;
            }
        }

        auto meshView = scene.registry.view<Mesh>();
        for ( auto id : meshView )
        {
            entt::handle entity = { scene.registry, id };

            if ( !entity.all_of<gl::MeshHolder>() )
                entity.emplace<gl::MeshHolder>( gl::MeshHolder::get( entity.get<Mesh>() ) );
        }

        auto entities = scene.registry.view<Molecule, MoleculeIDs, PyRepresentation>();
        for ( auto id : entities )
        {
            entt::handle entity                            = { scene.registry, id };
            auto [ molecule, moleculeIds, representation ] = entity.get<Molecule, MoleculeIDs, PyRepresentation>();

            switch ( representation.representation )
            {
            case RepresentationType::Ses:
            {
                if ( !entity.all_of<gl::SesdfHolder>() )
                    entity.emplace<gl::SesdfHolder>( gl::SesdfHolder::get( molecule ) );
                break;
            }
            case RepresentationType::vanDerWaals:
            {
                if ( !entity.all_of<gl::SphereHolder>() )
                {
                    entity.emplace<gl::SphereHolder>( gl::SphereHolder::getMolecule( molecule, &moleculeIds ) );
                }
                break;
            }
            case RepresentationType::BallAndStick:
            {
                if ( !entity.all_of<gl::BallAndStickHolder>() )
                    entity.emplace<gl::BallAndStickHolder>(
                        gl::BallAndStickHolder::getMolecule( molecule, &moleculeIds ) );
                break;
            }
            case RepresentationType::Sticks:
            {
                if ( !entity.all_of<gl::SticksHolder>() )
                    entity.emplace<gl::SticksHolder>( gl::SticksHolder::getMolecule( molecule, &moleculeIds ) );
                break;
            }
            case RepresentationType::Sas:
            {
                if ( !entity.all_of<gl::SasHolder>() )
                    entity.emplace<gl::SasHolder>( gl::SasHolder::getMolecule( molecule, &moleculeIds ) );
                break;
            }
            }
        }

        glViewport( 0, 0, m_width, m_height );
        gBufferPass.render(
            [ this, &scene, &cam ]
            {
                if ( m_geometryHandler )
                    m_geometryHandler->render( *cam, scene );
            } );
        postProcessPass.render( *cam );

        if ( m_window != nullptr )
        {
            SDL_Window * handle = m_window->getHandle();

            SDL_GL_MakeCurrent( handle, m_window->getContext() );
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
            glBindFramebuffer( GL_READ_FRAMEBUFFER, postProcessPass.getFramebuffer() );
            glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 );
            glBlitFramebuffer( 0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST );

            SDL_GL_SwapWindow( handle );
        }

        m_currentFBO = postProcessPass.getFramebuffer();
    }

    bool PyEngine::update( PyScene & scene, PyCamera * camera )
    {
        PyCamera * cam = camera != nullptr ? camera : scene.mainCamera;

        if ( cam == nullptr )
            throw std::runtime_error( "No camera provided in engine update call and scene main camera is null!" );

        render( scene, cam );

        if ( cam->controllerForwarder == nullptr )
        {
            cam->controllerForwarder = std::make_unique<ControllerForwarder>();
            cam->cameraController    = &cam->controllerForwarder->add<CameraController>( cam->self );
        }

        const Input & inputs = m_window->getInput();
        if ( inputs.windowResized )
        {
            cam->viewport = inputs.windowSize;
            resize( inputs.windowSize.x, inputs.windowSize.y );
        }

        if ( controlsEnabled )
        {
            cam->cameraController->type = cam->controller;
            cam->controllerForwarder->update( inputs );
        }

        return m_window->update();
    }

    std::vector<unsigned char> PyEngine::screenshot( PyScene & scene, const PyCamera * camera )
    {
        const PyCamera * cam = camera != nullptr ? camera : scene.mainCamera;

        if ( cam == nullptr )
            throw std::runtime_error( "No camera provided in engine screenshot call and scene main camera is null!" );

        render( scene, cam );

        auto image = std::vector<uint8_t>( m_width * m_height * 4 );
        glBindFramebuffer( GL_FRAMEBUFFER, getFramebuffer() );
        glReadnPixels(
            0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, static_cast<GLsizei>( image.size() ), image.data() );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        stbi_write_png_compression_level = 0;
        stbi_flip_vertically_on_write( true );

        int             len;
        unsigned char * imgPtr = stbi_write_png_to_mem( image.data(), 0, m_width, m_height, 4, &len );
        if ( imgPtr == nullptr )
            return {};

        std::vector<unsigned char> imgVector { imgPtr, imgPtr + len };

        STBIW_FREE( imgPtr );

        return imgVector;
    }

    void PyEngine::resize( uint32_t width, uint32_t height )
    {
        if ( m_width == width && m_height == height )
            return;

        m_width  = width;
        m_height = height;

        gBufferPass.resize( m_width, m_height );
        postProcessPass.resize( m_width, m_height );

        resizeRendererTexture();
    }

    inline GLuint PyEngine::getFramebuffer() const { return m_currentFBO; }

    inline gl::PostProcessPass & PyEngine::getPostProcess() { return postProcessPass; }

    std::vector<uint32_t> PyEngine::getIDsImage() const
    {
        auto image = std::vector<uint32_t>( m_width * m_height );
        glGetTextureImage( gBufferPass.getIdsTexture(),
                           0,
                           GL_RED_INTEGER,
                           GL_UNSIGNED_INT,
                           static_cast<GLsizei>( image.size() * sizeof( uint32_t ) ),
                           image.data() );
        return image;
    }
    std::vector<float> PyEngine::getDepthImage() const
    {
        auto image = std::vector<float>( m_width * m_height );
        glGetTextureImage( postProcessPass.getLinearizedDepthTexture(),
                           0,
                           GL_RED,
                           GL_FLOAT,
                           static_cast<GLsizei>( image.size() * sizeof( float ) ),
                           image.data() );
        return image;
    }
    std::vector<glm::vec4> PyEngine::getShadingImage() const
    {
        auto image = std::vector<glm::vec4>( m_width * m_height );
        glGetTextureImage( postProcessPass.getShadingTexture(),
                           0,
                           GL_RGBA,
                           GL_FLOAT,
                           static_cast<GLsizei>( image.size() * sizeof( glm::vec4 ) ),
                           image.data() );
        return image;
    }
    std::vector<glm::vec4> PyEngine::getMaterialImage() const
    {
        auto image = std::vector<glm::vec4>( m_width * m_height );
        glGetTextureImage( gBufferPass.getMaterialTexture(),
                           0,
                           GL_RGBA,
                           GL_FLOAT,
                           static_cast<GLsizei>( image.size() * sizeof( glm::vec4 ) ),
                           image.data() );
        return image;
    }

    void PyEngine::resizeRendererTexture()
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
} // namespace rvtx
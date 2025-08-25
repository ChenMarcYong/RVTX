#include <GL/gl3w.h>
#include <entt/entity/registry.hpp>
#include <fmt/chrono.h> // Used to handle 'std::tm' in fmt::format
#include <glm/gtc/type_ptr.hpp>
#include <rvtx/core/image.hpp>
#include <rvtx/optix/environment.hpp>

#include "rvtx/core/logger.hpp"
#include "rvtx/core/time.hpp"
#include "rvtx/cuda/buffer.cuh"
#include "rvtx/cuda/gl_interop/framebuffer.cuh"
#include "rvtx/gl/renderer.hpp"
#include "rvtx/gl/utils/program.hpp"
#include "rvtx/gl/utils/snapshot.hpp"
#include "rvtx/gl/window.hpp"
#include "rvtx/molecule/loader.hpp"
#include "rvtx/optix/context.cuh"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.hpp"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_handler.hpp"
#include "rvtx/optix/geometry/ses/ses_geometry.hpp"
#include "rvtx/optix/geometry/ses/ses_handler.hpp"
#include "rvtx/optix/geometry/sphere/sphere_geometry.hpp"
#include "rvtx/optix/geometry/sphere/sphere_handler.hpp"
#include "rvtx/system/camera.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/scene_descriptor.hpp"
#include "rvtx/ux/camera_controller.hpp"
#include "rvtx/ux/controller.hpp"

void updateCameraParameters( const rvtx::Camera & camera, rvtx::optix::PathTracerCamera & pathTracerCamera )
{
    const glm::mat4 view = glm::inverse( camera.getViewMatrix() );
    std::memcpy( pathTracerCamera.view, glm::value_ptr( view ), sizeof( float ) * 16 );

    const glm::mat4 projection = camera.getPerspectiveProjectionMatrix();
    std::memcpy( pathTracerCamera.projection, glm::value_ptr( projection ), sizeof( float ) * 16 );

    // For orthographic camera
    pathTracerCamera.distance = camera.target.distance;

    pathTracerCamera.isPerspective = camera.isPerspective();
}

int main( int /* argc */, char ** /* argv */ )
{
    rvtx::logger::info( "Started" );

    // Loads the saved scene
    rvtx::SceneDescriptor sceneDescriptor = rvtx::parse( "data/scene_1AGA.json" );

    rvtx::WindowDescriptor & wd = sceneDescriptor.windowDescriptor;
    rvtx::gl::Window         window { wd.title, wd.width, wd.height, wd.shown };

    // Initializes the GL renderer
    rvtx::RendererDescriptor & rd = sceneDescriptor.rendererDescriptor;
    rvtx::gl::ProgramManager   programManager { "shaders" };
    glm::uvec2                 viewport = rd.useWindowViewport ? wd.viewport : rd.viewport;
    rvtx::gl::Renderer         renderer { window, programManager, viewport };
    rvtx::optix::Context       context {};

    // Creates the scene and loads a molecule into it
    rvtx::Scene scene {};

    // Initializes the camera
    rvtx::CameraDescriptor & cd              = sceneDescriptor.cameraDescriptor;
    entt::handle             cameraEntity    = scene.createEntity( "Main Camera" );
    rvtx::Transform &        cameraTransform = cameraEntity.emplace<rvtx::Transform>( cd.transform );
    rvtx::Camera &           camera          = cameraEntity.emplace<rvtx::Camera>(
        cameraTransform, viewport, cd.target, cd.projectionType, glm::radians( cd.fov ) );

    // Adds camera controller(s)
    rvtx::ControllerForwarder forwarder {};
    auto &                    cameraController = forwarder.add<rvtx::CameraController>( cameraEntity );
    cameraController.setType( rvtx::CameraController::Type::Trackball );

    // Only load first entity
    for ( std::size_t i = 0; i < sceneDescriptor.entities.size(); i++ )
    {
        const auto &       entityDescriptor = sceneDescriptor.entities[ i ];
        const entt::handle entity           = scene.createEntity();
        auto &             transform        = entity.emplace<rvtx::Transform>( entityDescriptor.transform );

        switch ( entityDescriptor.type )
        {
        case rvtx::EntityDescriptor::Molecule:
        {
            auto & molecule = entity.emplace<rvtx::Molecule>( rvtx::load( entityDescriptor.path ) );
            molecule.aabb.attachTransform( &transform );

            // Each molecule can have multiple defined representations, they are loaded in this loop
            for ( const auto & representation : entityDescriptor.representations )
            {
                switch ( representation.type )
                {
                case rvtx::RepresentationType::Ses:
                {
                    auto & sesGeometry
                        = entity.emplace<rvtx::optix::SesGeometry>( context, molecule, representation.probeRadius );
                    sesGeometry.setColorMode( representation.colorMode );
                    sesGeometry.createCustomMaterial( representation.materialParameters );
                    break;
                }
                case rvtx::RepresentationType::BallAndStick:
                {
                    auto & basGeometry = entity.emplace<rvtx::optix::BallAndStickGeometry>( context, molecule );
                    basGeometry.setColorMode( representation.colorMode );
                    basGeometry.createCustomMaterial( representation.materialParameters );
                    break;
                }
                case rvtx::RepresentationType::vanDerWaals:
                {
                    auto & sphereGeometry = entity.emplace<rvtx::optix::SphereGeometry>( context, molecule );
                    sphereGeometry.setColorMode( representation.colorMode );
                    sphereGeometry.createCustomMaterial( representation.materialParameters );

                    break;
                }
                }
            }

            if ( cd.targetEntity == i )
                camera.target = rvtx::Camera::Target( molecule.getAabb() );

            break;
        }
        default: break;
        }
    }

    // Initialize OptiX pipelines
    rvtx::optix::MultiGeometryPipeline pipeline { context };
    pipeline.setPrimitiveType(
        static_cast<uint32_t>( OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE ) );

    rvtx::optix::Module rayGen { context, "ptx/path_tracer.ptx" };
    pipeline.setRayGen( rayGen, "__raygen__rg" );
    pipeline.setMiss( rayGen, "__miss__general" );

    pipeline.add<rvtx::optix::SesHandler>( pipeline, scene );
    pipeline.add<rvtx::optix::SphereHandler>( pipeline, scene );
    pipeline.add<rvtx::optix::BallAndStickHandler>( pipeline, scene );

    pipeline.compile();
    pipeline.updateGeometry();

    // Initializes cuda frame-buffer
    auto frameBuffer
        = rvtx::cuda::DeviceFrameBuffer::Typed<uchar4>( window.getWidth(), window.getHeight(), context.getStream() );
    auto accumulation = rvtx::cuda::DeviceBuffer::Typed<float4>( window.getWidth() * window.getHeight() );
    auto dParameters  = rvtx::cuda::DeviceBuffer::Typed<rvtx::optix::PathTracerParameters>( 1 );

    // Load the environment
    rvtx::BackgroundDescriptor & bd = sceneDescriptor.backgroundDescriptor;
    rvtx::optix::Environment     environment( bd.exrPath, bd.weight );

    // Prepare the path tracer parameters
    rvtx::optix::PathTracerParameters parameters {};
    parameters.environment         = environment.getEnvironmentView();
    parameters.environmentSampling = environment.getSamplingView();

    updateCameraParameters( camera, parameters.camera );

    parameters.bounces      = rd.maxRayBounces;
    parameters.pixelWidth   = rd.raysPerPixel;
    parameters.depthOfField = { cd.depthOfField.x, cd.depthOfField.y, cd.depthOfField.z };

    // Render loop
    uint32_t   subFrameId = 0;
    const auto sbt        = pipeline.getBindingTable();
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            // Prepare the parameters for the OptiX render
            parameters.handle       = pipeline.getHandle();
            parameters.accumulation = accumulation.get<float4>();
            parameters.frame        = frameBuffer.map<uchar4>();

            parameters.subFrameId = subFrameId++;
            parameters.viewSize   = make_uint2( frameBuffer.getWidth(), frameBuffer.getHeight() );

            rvtx::cuda::cudaCheck( cudaMemcpyAsync( dParameters.get(),
                                                    &parameters,
                                                    sizeof( rvtx::optix::PathTracerParameters ),
                                                    cudaMemcpyHostToDevice,
                                                    context.getStream() ) );

            // Render the scene to the frame-buffer
            pipeline.launch( dParameters.get(),
                             sizeof( rvtx::optix::PathTracerParameters ),
                             sbt,
                             frameBuffer.getWidth(),
                             frameBuffer.getHeight(),
                             1 );
            frameBuffer.unmap();

            glBindTexture( GL_TEXTURE_2D, renderer.getRendererTextureHandle() );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, frameBuffer.getId() );
            glTexSubImage2D( GL_TEXTURE_2D,
                             0,
                             0,
                             0,
                             frameBuffer.getWidth(),
                             frameBuffer.getHeight(),
                             GL_RGBA,
                             GL_UNSIGNED_BYTE,
                             nullptr );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

            // Render the frame-buffer texture using the GL renderer
            renderer.render();

            const rvtx::Input & inputs = window.getInput();
            if ( inputs.windowResized )
            {
                camera.viewport = inputs.windowSize;
                renderer.resize( inputs.windowSize.x, inputs.windowSize.y );
                frameBuffer.resize<uchar4>( inputs.windowSize.x, inputs.windowSize.y );
                renderer.resize( inputs.windowSize.x, inputs.windowSize.y );
                accumulation = rvtx::cuda::DeviceBuffer::Typed<float4>( inputs.windowSize.x * inputs.windowSize.y );
            }

            // Takes a snapshot
            if ( inputs.isKeyDown( rvtx::Key::F7 ) )
            {
                constexpr std::string_view SnapshotsDirectory = "snapshots";
                if ( !std::filesystem::is_directory( SnapshotsDirectory ) )
                    std::filesystem::create_directory( SnapshotsDirectory );
                rvtx::gl::snapshot( fmt::format( "{}/{:%Y-%m-%d_%H-%M-%S}.png", SnapshotsDirectory, rvtx::time::now() ),
                                    renderer.getFramebuffer(),
                                    window.getWidth(),
                                    window.getHeight() );
            }

            if ( inputs.isKeyDown( rvtx::Key::P ) )
            {
                camera.nextProjectionType();
                updateCameraParameters( camera, parameters.camera );
                subFrameId = 0;
            }

            if ( inputs.isKeyDown( rvtx::Key::B ) )
            {
                rvtx::logger::info( "{}\n{}", camera.transform->position, camera.transform->rotation );
            }

            // If any update was made, update the camera and accumulation
            if ( forwarder.update( inputs ) || inputs.windowResized )
            {
                updateCameraParameters( camera, parameters.camera );
                subFrameId = 0;
            }

            isRunning = window.update();
        }
    }
    catch ( const std::exception & e )
    {
        rvtx::logger::error( e.what() );
    }

    return EXIT_SUCCESS;
}

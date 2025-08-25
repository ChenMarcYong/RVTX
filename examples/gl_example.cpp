#include <entt/entity/registry.hpp>
#include <fmt/chrono.h> // Used to handle 'std::tm' in fmt::format
#include <rvtx/core/logger.hpp>
#include <rvtx/core/time.hpp>
#include <rvtx/gl/geometry/ball_and_stick_geometry.hpp>
#include <rvtx/gl/geometry/sphere_geometry.hpp>
#include <rvtx/gl/renderer.hpp>
#include <rvtx/gl/utils/program.hpp>
#include <rvtx/gl/utils/snapshot.hpp>
#include <rvtx/gl/window.hpp>
#include <rvtx/molecule/loader.hpp>
#include <rvtx/molecule/molecule.hpp>
#include <rvtx/system/camera.hpp>
#include <rvtx/system/name.hpp>
#include <rvtx/system/scene.hpp>
#include <rvtx/system/scene_descriptor.hpp>
#include <rvtx/system/transform.hpp>
#include <rvtx/ux/camera_controller.hpp>
#include <rvtx/ux/input.hpp>

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

    // Creates handled geometries of the GL renderer
    auto geometryForwarder = std::make_unique<rvtx::gl::GeometryForwarder>();
    geometryForwarder->add<rvtx::gl::SphereHandler>( programManager );
    geometryForwarder->add<rvtx::gl::BallAndStickHandler>( programManager );
    renderer.setGeometry( std::move( geometryForwarder ) );

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
                case rvtx::RepresentationType::vanDerWaals:
                    entity.emplace<rvtx::gl::SphereHolder>(
                        rvtx::gl::SphereHolder::getMolecule( molecule ) );
                    break;
                case rvtx::RepresentationType::BallAndStick:
                    entity.emplace<rvtx::gl::BallAndStickHolder>(
                        rvtx::gl::BallAndStickHolder::getMolecule( molecule ) );
                    break;
                default: throw std::runtime_error( "Unsupported molecule representation loaded." );
                }
            }

            if ( cd.targetEntity == i )
                camera.target = rvtx::Camera::Target( molecule.getAabb() );

            break;
        }
        default: break;
        }
    }

    // Render loop
    try
    {
        bool isRunning = true;
        while ( isRunning )
        {
            renderer.render( camera, scene );

            const rvtx::Input & inputs = window.getInput();
            if ( inputs.windowResized )
            {
                camera.viewport = inputs.windowSize;
                renderer.resize( inputs.windowSize.x, inputs.windowSize.y );
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
                camera.nextProjectionType();

            // Refreshes all shaders
            if ( inputs.isKeyDown( rvtx::Key::F8 ) )
                programManager.refresh();

            forwarder.update( inputs );
            isRunning = window.update();
        }
    }
    catch ( const std::exception & e )
    {
        rvtx::logger::error( e.what() );
    }

    return EXIT_SUCCESS;
}

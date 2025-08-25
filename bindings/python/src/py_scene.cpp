#include "pyrvtx/py_scene.hpp"

#include <rvtx/molecule/loader.hpp>
#include <rvtx/system/molecule_ids.hpp>
#include <rvtx/system/visibility.hpp>

#include "pyrvtx/py_mesh.hpp"
#include "rvtx/mesh/loader.hpp"

#if RVTX_GL
#include "pyrvtx/py_engine.hpp"
#endif

namespace rvtx
{
#if RVTX_GL
    bool PyScene::hasPyEngine() const
    {
        auto pyEngineView = registry.view<PyEngineView>();
        return pyEngineView.begin() != pyEngineView.end();
    }

    PyEngine * PyScene::getPyEngine() const
    {
        auto pyEngineView = registry.view<PyEngineView>();
        if ( pyEngineView.begin() != pyEngineView.end() )
        {
            // Should only have one pyEngine, returning the first found
            for ( auto entity : pyEngineView )
                return registry.get<PyEngineView>( entity ).pyengine;
        }
        return nullptr;
    }
#endif

    PyMolecule PyScene::loadMolecule( const std::filesystem::path & path, const RepresentationType representationType )
    {
        return createMolecule( rvtx::load( path ), representationType );
    }

    PyMesh PyScene::loadMesh( const std::filesystem::path & path )
    {
        const entt::handle entity    = createEntity( path.filename().string() );
        auto &             mesh      = entity.emplace<Mesh>( rvtx::loadMesh( path ) );
        auto &             transform = entity.emplace<Transform>();

        PyMesh pyMesh {};

        pyMesh.vertices = &mesh.vertices;
        pyMesh.ids      = &mesh.ids;
        pyMesh.indices  = &mesh.indices;
        pyMesh.aabb     = &mesh.aabb;

        pyMesh.transform = &transform;
        pyMesh.visible   = &registry.get<Visibility>( entity ).visible;
        pyMesh.self      = entity;

        pyMesh.aabb->attachTransform( pyMesh.transform );

        return pyMesh;
    }

    PyMolecule PyScene::createMolecule( Molecule inputMolecule, const RepresentationType representationType )
    {
        const entt::handle entity         = createEntity( inputMolecule.id );
        auto &             molecule       = entity.emplace<Molecule>( inputMolecule );
        auto &             moleculeIds    = entity.emplace<MoleculeIDs>( molecule );
        auto &             transform      = entity.emplace<Transform>();
        auto &             representation = entity.emplace<PyRepresentation>( representationType );

        PyMolecule pyMolecule {};

        pyMolecule.id            = &molecule.id;
        pyMolecule.name          = &molecule.name;
        pyMolecule.data          = &molecule.data;
        pyMolecule.atoms         = &molecule.atoms;
        pyMolecule.bonds         = &molecule.bonds;
        pyMolecule.residues      = &molecule.residues;
        pyMolecule.chains        = &molecule.chains;
        pyMolecule.peptideBonds  = &molecule.peptideBonds;
        pyMolecule.residentAtoms = &molecule.residentAtoms;
        pyMolecule.aabb          = &molecule.aabb;
        pyMolecule.ids           = &moleculeIds;

        pyMolecule.transform      = &transform;
        pyMolecule.representation = &representation;
        pyMolecule.visible        = &registry.get<Visibility>( entity ).visible;
        pyMolecule.self           = entity;

        pyMolecule.aabb->attachTransform( pyMolecule.transform );

        return pyMolecule;
    }

    PyMolecule PyScene::createProceduralMolecule( const ProceduralMoleculeGenerator & generator )
    {
        return createMolecule( generator.generate(), RepresentationType::vanDerWaals );
    }

    PyCamera PyScene::createCamera( const Transform &            transform,
                                    const Camera::Target         target,
                                    const Camera::Projection     projectionType,
                                    const CameraController::Type controller,
                                    const glm::uvec2             viewport )
    {
        entt::handle cameraEntity    = createEntity( "Main Camera" );
        Transform &  cameraTransform = cameraEntity.emplace<Transform>( transform );

        PyCamera camera;

        camera.transform      = &cameraTransform;
        camera.viewport       = viewport;
        camera.target         = target;
        camera.projectionType = projectionType;

        camera.self = cameraEntity;

        camera.controller = controller;

        cameraEntity.emplace<Camera>( camera );

        return camera;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec3> & points,
                                                const std::vector<glm::vec3> & colors,
                                                const std::vector<float> &     radii )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points.reserve( points.size() );
        for ( std::size_t i = 0; i < glm::min( radii.size(), points.size() ); i++ )
            pointCloud.points.emplace_back( points[ i ], radii[ i ] );
        for ( std::size_t i = radii.size(); i < points.size(); i++ )
            pointCloud.points.emplace_back( points[ i ], radii[ 0 ] );

        pointCloud.pointsColors.reserve( glm::min( colors.size(), points.size() ) );
        for ( std::size_t i = 0; i < glm::min( colors.size(), points.size() ); i++ )
            pointCloud.pointsColors.emplace_back( colors[ i ], 1.f );
        pointCloud.pointsColors.resize( points.size(), glm::vec4 { colors[ 0 ], 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec3> & points,
                                                const glm::vec3 &              color,
                                                const float                    radius )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points.reserve( points.size() );
        for ( const glm::vec3 & point : points )
            pointCloud.points.emplace_back( point, radius );

        pointCloud.pointsColors.resize( points.size(), glm::vec4 { color, 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec3> & points,
                                                const std::vector<float> &     radii,
                                                const std::vector<glm::vec3> & colors )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points.reserve( points.size() );
        for ( std::size_t i = 0; i < glm::min( radii.size(), points.size() ); i++ )
            pointCloud.points.emplace_back( points[ i ], radii[ i ] );
        for ( std::size_t i = radii.size(); i < points.size(); i++ )
            pointCloud.points.emplace_back( points[ i ], radii[ 0 ] );

        pointCloud.pointsColors.reserve( glm::min( colors.size(), points.size() ) );
        for ( std::size_t i = 0; i < glm::min( colors.size(), points.size() ); i++ )
            pointCloud.pointsColors.emplace_back( colors[ i ], 1.f );
        pointCloud.pointsColors.resize( points.size(), glm::vec4 { colors[ 0 ], 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec3> & points,
                                                const float                    radius,
                                                const glm::vec3 &              color )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points.reserve( points.size() );
        for ( const glm::vec3 & point : points )
            pointCloud.points.emplace_back( point, radius );

        pointCloud.pointsColors.resize( points.size(), glm::vec4 { color, 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec4> & points,
                                                const std::vector<glm::vec4> & colors )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points = points;

        pointCloud.pointsColors.reserve( glm::min( colors.size(), points.size() ) );
        for ( std::size_t i = 0; i < glm::min( colors.size(), points.size() ); i++ )
            pointCloud.pointsColors.emplace_back( colors[ i ] );
        pointCloud.pointsColors.resize( points.size(), colors[ 0 ] );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec4> & points,
                                                const std::vector<glm::vec3> & colors )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points = points;

        pointCloud.pointsColors.reserve( glm::min( colors.size(), points.size() ) );
        for ( std::size_t i = 0; i < glm::min( colors.size(), points.size() ); i++ )
            pointCloud.pointsColors.emplace_back( colors[ i ], 1.f );
        pointCloud.pointsColors.resize( points.size(), glm::vec4 { colors[ 0 ], 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec4> & points, const glm::vec4 & color )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points = points;

        pointCloud.pointsColors.resize( points.size(), color );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyPointCloudView PyScene::createPointCloud( const std::vector<glm::vec4> & points, const glm::vec3 & color )
    {
        if ( points.size() <= 0 )
            throw std::runtime_error( "The number of points must be greater than 0" );

        auto           pointCloudEntity = createEntity();
        PyPointCloud & pointCloud       = pointCloudEntity.emplace<PyPointCloud>();

        pointCloud.points = points;

        pointCloud.pointsColors.resize( points.size(), glm::vec4 { color, 1.f } );

        pointCloud.update();

        PyPointCloudView pointCloudView;

        pointCloudView.self         = pointCloudEntity;
        pointCloudView.points       = &pointCloud.points;
        pointCloudView.pointsColors = &pointCloud.pointsColors;
        pointCloudView.pointCloud   = &pointCloud;

        return pointCloudView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec4> & nodes,
                                      const std::vector<unsigned> &  edges,
                                      const std::vector<glm::vec4> & nodesColors,
                                      const std::vector<glm::vec4> & edgesData )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );
        if ( edges.size() < 2 )
            throw std::runtime_error( "The number of edges must be at least 1" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        // Nodes
        graph.nodes = nodes;

        // Nodes colors
        graph.nodesColors.reserve( glm::min( nodesColors.size(), nodes.size() ) );
        for ( std::size_t i = 0; i < glm::min( nodesColors.size(), nodes.size() ); i++ )
            graph.nodesColors.emplace_back( nodesColors[ i ] );
        graph.nodesColors.resize( nodes.size(), nodesColors[ 0 ] );

        // Edges
        graph.edges = edges;

        // Edges data
        graph.edgesData.reserve( glm::min( edgesData.size(), edges.size() / 2 ) );
        for ( std::size_t i = 0; i < glm::min( edgesData.size(), edges.size() / 2 ); i++ )
            graph.edgesData.emplace_back( edgesData[ i ] );
        graph.edgesData.resize( edges.size() / 2, edgesData[ 0 ] );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec3> & nodes,
                                      const std::vector<unsigned> &  edges,
                                      const std::vector<float> &     nodesRadii,
                                      const std::vector<float> &     edgesRadii,
                                      const std::vector<glm::vec3> & nodesColors,
                                      const std::vector<glm::vec3> & edgesColors )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );
        if ( edges.size() < 2 )
            throw std::runtime_error( "The number of edges must be at least 1" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        // Nodes
        graph.nodes.reserve( nodes.size() );
        for ( std::size_t i = 0; i < glm::min( nodesRadii.size(), nodes.size() ); i++ )
            graph.nodes.emplace_back( nodes[ i ], nodesRadii[ i ] );
        for ( std::size_t i = nodesRadii.size(); i < nodes.size(); i++ )
            graph.nodes.emplace_back( nodes[ i ], nodesRadii[ 0 ] );

        // Nodes colors
        graph.nodesColors.reserve( glm::min( nodesColors.size(), nodes.size() ) );
        for ( std::size_t i = 0; i < glm::min( nodesColors.size(), nodes.size() ); i++ )
            graph.nodesColors.emplace_back( nodesColors[ i ], 1.f );
        graph.nodesColors.resize( nodes.size(), glm::vec4 { nodesColors[ 0 ], 1.f } );

        // Edges
        graph.edges = edges;

        // Edges data
        graph.edgesData.reserve( glm::min( edges.size() / 2, glm::max( edgesRadii.size(), edgesColors.size() ) ) );
        for ( std::size_t i = 0; i < edges.size() / 2; i++ )
            graph.edgesData.emplace_back( i < edgesColors.size() ? edgesColors[ i ] : edgesColors[ 0 ],
                                          i < edgesRadii.size() ? edgesRadii[ i ] : edgesRadii[ 0 ] );

        graph.edgesData.resize( edges.size() / 2, graph.edgesData[ 0 ] );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec3> & nodes,
                                      const std::vector<unsigned> &  edges,
                                      const float                    nodesRadius,
                                      const float                    edgesRadius,
                                      const glm::vec3 &              nodesColor,
                                      const glm::vec3 &              edgesColor )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );
        if ( edges.size() < 2 )
            throw std::runtime_error( "The number of edges must be at least 1" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        // Nodes
        graph.nodes.reserve( nodes.size() );
        for ( int i = 0; i < nodes.size(); i++ )
            graph.nodes.emplace_back( nodes[ i ], nodesRadius );

        // Nodes colors
        graph.nodesColors.resize( nodes.size(), glm::vec4 { nodesColor, 1.f } );

        // Edges
        graph.edges = edges;

        // Edges data
        graph.edgesData.resize( edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec3> & nodes,
                                      const std::vector<unsigned> &  edges,
                                      const float                    edgesRadius,
                                      const glm::vec3 &              edgesColor )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );
        if ( edges.size() < 2 )
            throw std::runtime_error( "The number of edges must be at least 1" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        // Nodes
        graph.nodes.reserve( nodes.size() );
        for ( int i = 0; i < nodes.size(); i++ )
            graph.nodes.emplace_back( nodes[ i ], edgesRadius );

        // Nodes colors
        graph.nodesColors.resize( nodes.size(), glm::vec4 { edgesColor, 1.f } );

        // Edges
        graph.edges = edges;

        // Edges data
        graph.edgesData.resize( edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec4> & nodes,
                                      const float                    edgesRadius,
                                      const glm::vec3 &              edgesColor,
                                      const PyGraph::ConnectionType  connectionType )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes = nodes;
        graph.nodesColors.resize( nodes.size(), glm::vec4 { edgesColor, 1.f } );

        switch ( connectionType )
        {
        case PyGraph::ConnectionType::LINES:
        {
            for ( int i = 0; i < nodes.size() - 1; i += 2 )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_STRIP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_LOOP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            graph.edges.emplace_back( static_cast<unsigned>( nodes.size() - 1 ) );
            graph.edges.emplace_back( 0 );
            break;
        }
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec3> & nodes,
                                      const float                    edgesRadius,
                                      const glm::vec3 &              edgesColor,
                                      const PyGraph::ConnectionType  connectionType )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes.reserve( nodes.size() );
        for ( const glm::vec3 & node : nodes )
            graph.nodes.emplace_back( node, edgesRadius );

        graph.nodesColors.resize( nodes.size(), glm::vec4 { edgesColor, 1.f } );

        switch ( connectionType )
        {
        case PyGraph::ConnectionType::LINES:
        {
            for ( int i = 0; i < nodes.size() - 1; i += 2 )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_STRIP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_LOOP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            graph.edges.emplace_back( static_cast<unsigned>( nodes.size() - 1 ) );
            graph.edges.emplace_back( 0 );
            break;
        }
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec4> & nodes,
                                      const glm::vec3 &              edgesColor,
                                      const float                    edgesRadius,
                                      const PyGraph::ConnectionType  connectionType )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes = nodes;
        graph.nodesColors.resize( nodes.size(), glm::vec4 { edgesColor, 1.f } );

        switch ( connectionType )
        {
        case PyGraph::ConnectionType::LINES:
        {
            for ( int i = 0; i < nodes.size() - 1; i += 2 )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_STRIP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_LOOP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            graph.edges.emplace_back( static_cast<unsigned>( nodes.size() - 1 ) );
            graph.edges.emplace_back( 0 );
            break;
        }
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const std::vector<glm::vec3> & nodes,
                                      const glm::vec3 &              edgesColor,
                                      const float                    edgesRadius,
                                      const PyGraph::ConnectionType  connectionType )
    {
        if ( nodes.size() < 2 )
            throw std::runtime_error( "The number of nodes must be at least 2" );

        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes.reserve( nodes.size() );
        for ( const glm::vec3 & node : nodes )
            graph.nodes.emplace_back( node, edgesRadius );

        graph.nodesColors.resize( nodes.size(), glm::vec4 { edgesColor, 1.f } );

        switch ( connectionType )
        {
        case PyGraph::ConnectionType::LINES:
        {
            for ( int i = 0; i < nodes.size() - 1; i += 2 )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_STRIP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            break;
        }
        case PyGraph::ConnectionType::LINE_LOOP:
        {
            for ( int i = 0; i < nodes.size() - 1; i++ )
            {
                graph.edges.emplace_back( i );
                graph.edges.emplace_back( i + 1 );
            }
            graph.edges.emplace_back( static_cast<unsigned>( nodes.size() - 1 ) );
            graph.edges.emplace_back( 0 );
            break;
        }
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }

    PyGraphView PyScene::createGraph( const Path<glm::vec3> & path,
                                      const uint32_t          numPoints,
                                      const glm::vec3 &       edgesColor,
                                      const float             edgesRadius,
                                      bool                    showKey,
                                      const glm::vec3 &       keyColors,
                                      const float             keyRadius )
    {
        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes.reserve( numPoints );
        for ( std::size_t i = 0; i < numPoints; ++i )
        {
            graph.nodes.emplace_back( path.at( i * path.getDuration() / ( numPoints - 1 ) ), edgesRadius );
            graph.nodesColors.emplace_back( edgesColor, 1.f );
        }

        for ( int i = 0; i < graph.nodes.size() - 1; i++ )
        {
            graph.edges.emplace_back( i );
            graph.edges.emplace_back( i + 1 );
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        if ( showKey )
        {
            for ( const glm::vec3 & key : path.getValues() )
            {
                graph.nodes.emplace_back( key, keyRadius );
                graph.nodesColors.emplace_back( keyColors, 1.f );
            }
        }

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }
    PyGraphView PyScene::createGraph( const Mesh &      mesh,
                                      const glm::vec3 & edgesColor,
                                      const float       edgesRadius,
                                      const glm::mat4 & transform )
    {
        auto      graphEntity = createEntity();
        PyGraph & graph       = graphEntity.emplace<PyGraph>();

        graph.nodes.reserve( mesh.vertices.size() );

        for ( std::size_t i = 0; i < mesh.vertices.size(); ++i )
        {
            graph.nodes.emplace_back( glm::vec3 { transform * mesh.vertices[ i ].position }, edgesRadius );
            graph.nodesColors.emplace_back( edgesColor, 1.f );
        }

        graph.edges.reserve( mesh.indices.size() * 2 );
        for ( int i = 0; i < mesh.indices.size(); i += 3 )
        {
            graph.edges.emplace_back( mesh.indices[ i ] );
            graph.edges.emplace_back( mesh.indices[ i + 1 ] );

            graph.edges.emplace_back( mesh.indices[ i ] );
            graph.edges.emplace_back( mesh.indices[ i + 2 ] );

            graph.edges.emplace_back( mesh.indices[ i + 1 ] );
            graph.edges.emplace_back( mesh.indices[ i + 2 ] );
        }

        graph.edgesData.resize( graph.edges.size() / 2, glm::vec4 { edgesColor, edgesRadius } );

        graph.update();

        PyGraphView graphView;

        graphView.self        = graphEntity;
        graphView.nodes       = &graph.nodes;
        graphView.nodesColors = &graph.nodesColors;
        graphView.edges       = &graph.edges;
        graphView.edgesData   = &graph.edgesData;
        graphView.graph       = &graph;

        return graphView;
    }
} // namespace rvtx

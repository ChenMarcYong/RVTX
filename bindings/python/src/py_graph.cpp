#include "pyrvtx/py_graph.hpp"

#include "pyrvtx/py_scene.hpp"

namespace rvtx
{
    PyGraphView::~PyGraphView()
    {
        if ( scene != nullptr && scene->registry.valid( self ) )
        {
            scene->registry.destroy( self );
        }
    }

    PyGraphView::PyGraphView( PyGraphView && other ) noexcept
    {
        nodes       = std::exchange( other.nodes, {} );
        nodesColors = std::exchange( other.nodesColors, {} );
        edges       = std::exchange( other.edges, {} );
        edgesData   = std::exchange( other.edgesData, {} );
        self        = std::exchange( other.self, entt::handle {} );
        scene       = std::exchange( other.scene, nullptr );
    }

    PyGraphView & PyGraphView::operator=( PyGraphView && other ) noexcept
    {
        std::swap( nodes, other.nodes );
        std::swap( nodesColors, other.nodesColors );
        std::swap( edges, other.edges );
        std::swap( edgesData, other.edgesData );
        std::swap( self, other.self );
        std::swap( scene, other.scene );

        return *this;
    }

    PyGraph::PyGraph( PyGraph && other ) noexcept
    {
        nodes       = std::exchange( other.nodes, {} );
        nodesColors = std::exchange( other.nodesColors, {} );
        edges       = std::exchange( other.edges, {} );
        edgesData   = std::exchange( other.edgesData, {} );
#if RVTX_GL
        needsUpdate = std::exchange( other.needsUpdate, false );
        holder      = std::exchange( other.holder, nullptr );
#endif
    }

    PyGraph & PyGraph::operator=( PyGraph && other ) noexcept
    {
        std::swap( nodes, other.nodes );
        std::swap( nodesColors, other.nodesColors );
        std::swap( edges, other.edges );
        std::swap( edgesData, other.edgesData );
#if RVTX_GL
        std::swap( needsUpdate, other.needsUpdate );
        std::swap( holder, other.holder );
#endif

        return *this;
    }

    void PyGraph::update()
    {
#if RVTX_GL
        if ( holder == nullptr )
        {
            needsUpdate = true;
            return;
        }

        holder->nodesCount = static_cast<uint32_t>( nodes.size() );
        if ( holder->nodesCount > 0 )
        {
            holder->nodesBuffer       = gl::Buffer::Typed<glm::vec4>( nodes );
            holder->nodesColorsBuffer = gl::Buffer::Typed<glm::vec4>( nodesColors );
        }

        holder->edgesCount = static_cast<uint32_t>( edges.size() );
        if ( holder->edgesCount > 0 )
        {
            holder->edgesBuffer       = gl::Buffer::Typed<unsigned>( edges );
            holder->edgesParamsBuffer = gl::Buffer::Typed<glm::vec4>( edgesData );
        }

        needsUpdate = false;
#endif
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec4> & nodes,
                                          const std::vector<unsigned> &  edges,
                                          const std::vector<glm::vec4> & nodesColor,
                                          const std::vector<glm::vec4> & edgesData,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edges, nodesColor, edgesData );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec3> & nodes,
                                          const std::vector<unsigned> &  edges,
                                          const std::vector<float> &     nodesRadii,
                                          const std::vector<float> &     edgesRadii,
                                          const std::vector<glm::vec3> & nodesColors,
                                          const std::vector<glm::vec3> & edgesColors,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edges, nodesRadii, edgesRadii, nodesColors, edgesColors );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec3> & nodes,
                                          const std::vector<unsigned> &  edges,
                                          const float                    nodesRadius,
                                          const float                    edgesRadius,
                                          const glm::vec3 &              nodesColor,
                                          const glm::vec3 &              edgesColor,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edges, nodesRadius, edgesRadius, nodesColor, edgesColor );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec3> & nodes,
                                          const std::vector<unsigned> &  edges,
                                          const float                    edgesRadius,
                                          const glm::vec3 &              edgesColor,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edges, edgesRadius, edgesColor );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec4> & nodes,
                                          const float                    edgesRadius,
                                          const glm::vec3 &              edgesColor,
                                          const PyGraph::ConnectionType  connectionType,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edgesRadius, edgesColor, connectionType );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec3> & nodes,
                                          const float                    edgesRadius,
                                          const glm::vec3 &              edgesColor,
                                          const PyGraph::ConnectionType  connectionType,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edgesRadius, edgesColor, connectionType );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec4> & nodes,
                                          const glm::vec3 &              edgesColor,
                                          const float                    edgesRadius,
                                          const PyGraph::ConnectionType  connectionType,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edgesRadius, edgesColor, connectionType );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const Path<glm::vec3> & path,
                                          const uint32_t          numPoints,
                                          const glm::vec3 &       edgesColor,
                                          const float             edgesRadius,
                                          bool                    showKey,
                                          const glm::vec3 &       keyColors,
                                          const float             keyRadius,
                                          PyScene &               scene )
    {
        PyGraphView view = scene.createGraph( path, numPoints, edgesColor, edgesRadius, showKey, keyColors, keyRadius );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    PyGraphView PyGraphView::createGraph( const std::vector<glm::vec3> & nodes,
                                          const glm::vec3 &              edgesColor,
                                          const float                    edgesRadius,
                                          const PyGraph::ConnectionType  connectionType,
                                          PyScene &                      scene )
    {
        PyGraphView view = scene.createGraph( nodes, edgesRadius, edgesColor, connectionType );

        view.scene = &scene;

#if RVTX_GL
        if ( scene.hasPyEngine() )
        {
            view.graph->holder = &view.self.emplace<gl::DebugPrimitivesHolder>();
            view.update();
        }
#endif

        return view;
    }

    void PyGraphView::update() { graph->update(); }
} // namespace rvtx

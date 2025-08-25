#ifndef PYRVTX_PY_GRAPH_HPP
#define PYRVTX_PY_GRAPH_HPP

#if RVTX_GL
#include "rvtx/gl/system/debug_primitives.hpp"
#endif

namespace rvtx
{
    class PyScene;

    struct PyGraph
    {
        enum ConnectionType
        {
            LINES,
            LINE_STRIP,
            LINE_LOOP,
        };

        static constexpr auto in_place_delete = true;

        PyGraph()  = default;
        ~PyGraph() = default;

        PyGraph( const PyGraph & )             = delete;
        PyGraph & operator=( const PyGraph & ) = delete;

        PyGraph( PyGraph && other ) noexcept;
        PyGraph & operator=( PyGraph && other ) noexcept;

        std::vector<glm::vec4>    nodes;
        std::vector<glm::vec4>    nodesColors;
        std::vector<unsigned int> edges;
        std::vector<glm::vec4>    edgesData;

        bool needsUpdate = false;
        void update();

#if RVTX_GL
        gl::DebugPrimitivesHolder * holder;
#endif
    };

    struct PyGraphView
    {
        PyGraphView() = default;
        ~PyGraphView();

        PyGraphView( const PyGraphView & )             = delete;
        PyGraphView & operator=( const PyGraphView & ) = delete;

        PyGraphView( PyGraphView && other ) noexcept;
        PyGraphView & operator=( PyGraphView && other ) noexcept;

        entt::handle self;
        PyScene *    scene { nullptr };

        std::vector<glm::vec4> *    nodes;
        std::vector<glm::vec4> *    nodesColors;
        std::vector<unsigned int> * edges;
        std::vector<glm::vec4> *    edgesData;

        PyGraph * graph { nullptr };

        static PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                        const std::vector<unsigned> &  edges,
                                        const std::vector<glm::vec4> & nodesColors,
                                        const std::vector<glm::vec4> & edgesData,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                        const std::vector<unsigned> &  edges,
                                        const std::vector<float> &     nodesRadii,
                                        const std::vector<float> &     edgesRadii,
                                        const std::vector<glm::vec3> & nodesColors,
                                        const std::vector<glm::vec3> & edgesColors,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                        const std::vector<unsigned> &  edges,
                                        const float                    nodesRadius,
                                        const float                    edgesRadius,
                                        const glm::vec3 &              nodesColor,
                                        const glm::vec3 &              edgesColor,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                        const std::vector<unsigned> &  edges,
                                        const float                    edgesRadius,
                                        const glm::vec3 &              edgesColor,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                        const float                    edgesRadius,
                                        const glm::vec3 &              edgesColor,
                                        const PyGraph::ConnectionType  connectionType,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                        const float                    edgesRadius,
                                        const glm::vec3 &              edgesColor,
                                        const PyGraph::ConnectionType  connectionType,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec4> & nodes,
                                        const glm::vec3 &              edgesColor,
                                        const float                    edgesRadius,
                                        const PyGraph::ConnectionType  connectionType,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const std::vector<glm::vec3> & nodes,
                                        const glm::vec3 &              edgesColor,
                                        const float                    edgesRadius,
                                        const PyGraph::ConnectionType  connectionType,
                                        PyScene &                      scene );

        static PyGraphView createGraph( const Path<glm::vec3> & path,
                                        const uint32_t          numPoints,
                                        const glm::vec3 &       edgesColor,
                                        const float             edgesRadius,
                                        bool                    showKey,
                                        const glm::vec3 &       keyColors,
                                        const float             keyRadius,
                                        PyScene &               scene );

        void update();
    };
} // namespace rvtx

#endif
#ifndef RVTX_GL_GEOMETRY_DEBUG_PRIMITIVES_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_DEBUG_PRIMITIVES_GEOMETRY_HPP

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"

namespace rvtx::gl
{
    struct EdgeParams;
    struct NodeParams;

    struct DebugPrimitivesHolder
    {
        DebugPrimitivesHolder()  = default;
        ~DebugPrimitivesHolder() = default;

        DebugPrimitivesHolder( const DebugPrimitivesHolder & )             = delete;
        DebugPrimitivesHolder & operator=( const DebugPrimitivesHolder & ) = delete;

        DebugPrimitivesHolder( DebugPrimitivesHolder && ) noexcept             = default;
        DebugPrimitivesHolder & operator=( DebugPrimitivesHolder && ) noexcept = default;

        void setNodesColor( const glm::vec4 color = glm::vec4 { 1.0f } );
        void setNodesColor( const glm::vec3 color = glm::vec3 { 1.0f }, const bool visible = true );
        void setEdgesParams( const glm::vec3 color = glm::vec3 { 1.0f }, const float radius = 1.f );

        static DebugPrimitivesHolder get( const std::vector<glm::vec4> & nodes,
                                          std::vector<glm::vec4>         nodesColors  = { { 1.f, 0.f, 0.f, 1.f } },
                                          const std::vector<uint32_t> &  edgesIndices = {},
                                          std::vector<glm::vec4>         edgesParams  = {} );

        uint32_t nodesCount = 0;
        Buffer   nodesBuffer; // Buffer should be 'nodesCount' 4-aligned floats with 'x, y, z' = position, 'w' = radius
        Buffer nodesColorsBuffer; // Buffer should be 'nodesCount' 4-aligned floats with 'x, y, z' = color, 'w' = alpha

        uint32_t edgesCount = 0;
        Buffer   edgesBuffer; // Buffer should be 'edgesCount' 4-bytes indices, each pointing to a 'nodesBuffer' element
        Buffer   edgesParamsBuffer; // Buffer should be half 'edgesCount' 4-aligned floats with 'x, y, z' = color, 'w' =
                                    // radius
    };

    class Program;
    class ProgramManager;
    class System;
    class DebugPrimitivesHandler : public GeometryHandler
    {
      public:
        DebugPrimitivesHandler() = default;
        DebugPrimitivesHandler( ProgramManager & manager );
        ~DebugPrimitivesHandler() override;

        DebugPrimitivesHandler( const DebugPrimitivesHandler & )             = delete;
        DebugPrimitivesHandler & operator=( const DebugPrimitivesHandler & ) = delete;

        DebugPrimitivesHandler( DebugPrimitivesHandler && ) noexcept;
        DebugPrimitivesHandler & operator=( DebugPrimitivesHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_nodesProgram = nullptr;
        Program *     m_edgesProgram = nullptr;
        UniformBuffer m_cameraUniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_DEBUG_PRIMITIVES_GEOMETRY_HPP

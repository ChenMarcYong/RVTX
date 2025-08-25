#ifndef RVTX_GL_GEOMETRY_MESH_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_MESH_GEOMETRY_HPP

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"
#include "rvtx/mesh/mesh.hpp"

namespace rvtx::gl
{
    struct MeshHolder
    {
        MeshHolder()  = default;
        ~MeshHolder() = default;

        MeshHolder( const MeshHolder & )             = delete;
        MeshHolder & operator=( const MeshHolder & ) = delete;

        MeshHolder( MeshHolder && ) noexcept             = default;
        MeshHolder & operator=( MeshHolder && ) noexcept = default;

        static MeshHolder get( const Mesh & mesh );

        uint32_t indicesCount = 0;
        Buffer   verticesBuffer;
        Buffer   indicesBuffer;
        Buffer   idsBuffer;
        bool     useSingleId = true;
    };

    class Program;
    class ProgramManager;
    class System;
    class MeshHandler : public GeometryHandler
    {
      public:
        MeshHandler() = default;
        MeshHandler( ProgramManager & manager );
        ~MeshHandler() override;

        MeshHandler( const MeshHandler & )             = delete;
        MeshHandler & operator=( const MeshHandler & ) = delete;

        MeshHandler( MeshHandler && ) noexcept;
        MeshHandler & operator=( MeshHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_program = nullptr;
        UniformBuffer m_uniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_MESH_GEOMETRY_HPP

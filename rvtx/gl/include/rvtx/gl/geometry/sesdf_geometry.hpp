#ifndef RVTX_GL_GEOMETRY_SESDF_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_SESDF_GEOMETRY_HPP

#include <bcs/sesdf/sesdf.hpp>

#include "rvtx/core/aabb.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/uniform.hpp"

namespace rvtx
{
    struct Molecule;
}

namespace rvtx::gl
{
    struct SesdfHolder
    {
        ~SesdfHolder();

        SesdfHolder( const SesdfHolder & )             = delete;
        SesdfHolder & operator=( const SesdfHolder & ) = delete;

        SesdfHolder( SesdfHolder && ) noexcept;
        SesdfHolder & operator=( SesdfHolder && ) noexcept;

        static SesdfHolder get( const Molecule & molecule, float probeRadius = 1.4f );
        void               update() const;

        Aabb                              aabb;
        mutable bcs::Sesdf                data;
        mutable bcs::sesdf::SesdfGraphics surface {};

        GLuint segmentVao = GL_INVALID_VALUE;
        GLuint circleVao  = GL_INVALID_VALUE;
        GLuint convexVao  = GL_INVALID_VALUE;

      private:
        SesdfHolder( const Molecule & molecule, float probeRadius = 1.4f );
    };

    class Program;
    class ProgramManager;
    class System;
    class SesdfHandler : public GeometryHandler
    {
      public:
        SesdfHandler() = default;
        SesdfHandler( ProgramManager & manager );
        ~SesdfHandler() override;

        SesdfHandler( const SesdfHandler & )             = delete;
        SesdfHandler & operator=( const SesdfHandler & ) = delete;

        SesdfHandler( SesdfHandler && ) noexcept;
        SesdfHandler & operator=( SesdfHandler && ) noexcept;


        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program * m_convexPatchProgram  = nullptr;
        Program * m_segmentPatchProgram = nullptr;
        Program * m_circlePatchProgram  = nullptr;
        Program * m_concavePatchProgram = nullptr;

        UniformBuffer m_uniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };

    class SesdfStructureHandler : public GeometryHandler
    {
      public:
        enum class Mode
        {
            Line,
            Surface
        };

        SesdfStructureHandler() = default;
        SesdfStructureHandler( ProgramManager & manager );

        SesdfStructureHandler( const SesdfStructureHandler & )             = delete;
        SesdfStructureHandler & operator=( const SesdfStructureHandler & ) = delete;

        SesdfStructureHandler( SesdfStructureHandler && ) noexcept;
        SesdfStructureHandler & operator=( SesdfStructureHandler && ) noexcept;

        ~SesdfStructureHandler();

        inline void setMode( Mode mode );
        void        render( const Camera & camera, const Scene & scene ) override;

      private:
        Mode m_mode = Mode::Line;

        Program * m_lineTetrahedronProgram    = nullptr;
        Program * m_surfaceTetrahedronProgram = nullptr;

        UniformBuffer m_uniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#include "rvtx/gl/geometry/sesdf_geometry.inl"

#endif // RVTX_GL_GEOMETRY_SESDF_GEOMETRY_HPP

#ifndef RVTX_GL_GEOMETRY_SSESDF_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_SSESDF_GEOMETRY_HPP

#include <bcs/ssesdf/ssesdf.hpp>

#include "rvtx/core/aabb.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/uniform.hpp"

namespace rvtx
{
    struct Molecule;
}

namespace rvtx::gl
{
    struct SsesdfHolder
    {
        ~SsesdfHolder();

        SsesdfHolder( const SsesdfHolder & )             = delete;
        SsesdfHolder & operator=( const SsesdfHolder & ) = delete;

        SsesdfHolder( SsesdfHolder && ) noexcept;
        SsesdfHolder & operator=( SsesdfHolder && ) noexcept;

        static SsesdfHolder get( const Molecule & molecule, float probeRadius = 1.4f );
        void                update() const;

        Aabb                                aabb;
        mutable bcs::Ssesdf                 data;
        mutable bcs::ssesdf::SsesdfGraphics surface {};

        GLuint atomVao              = GL_INVALID_VALUE;
        GLuint torusVao             = GL_INVALID_VALUE;
        GLuint sphericalTriangleVao = GL_INVALID_VALUE;

      private:
        SsesdfHolder( const Molecule & molecule, float probeRadius = 1.4f );
    };

    class Program;
    class ProgramManager;
    class System;
    class SsesdfHandler : public GeometryHandler
    {
      public:
        SsesdfHandler() = default;
        SsesdfHandler( ProgramManager & manager );
        ~SsesdfHandler() override = default;

        SsesdfHandler( const SsesdfHandler & )             = delete;
        SsesdfHandler & operator=( const SsesdfHandler & ) = delete;

        SsesdfHandler( SsesdfHandler && ) noexcept             = default;
        SsesdfHandler & operator=( SsesdfHandler && ) noexcept = default;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program * m_atomProgram              = nullptr;
        Program * m_sphericalTriangleProgram = nullptr;
        Program * m_torusProgram             = nullptr;

        UniformBuffer m_uniforms;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_SSESDF_GEOMETRY_HPP

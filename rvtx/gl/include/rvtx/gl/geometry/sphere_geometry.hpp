#ifndef RVTX_GL_GEOMETRY_SPHERE_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_SPHERE_GEOMETRY_HPP

#include <optional>

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"

namespace rvtx::gl
{
    struct SphereHolder
    {
         SphereHolder() = default;
        ~SphereHolder() = default;

                       SphereHolder( const SphereHolder & ) = delete;
        SphereHolder & operator=( const SphereHolder & )    = delete;

                       SphereHolder( SphereHolder && ) noexcept = default;
        SphereHolder & operator=( SphereHolder && ) noexcept    = default;

        static SphereHolder getMolecule( const Molecule & molecule, const MoleculeIDs * moleculeIds = nullptr );
        static SphereHolder getNonResident( const Molecule & molecule, const MoleculeIDs * moleculeIds = nullptr );
        static SphereHolder getSystem( const Molecule & molecule, const MoleculeIDs * moleculeIds = nullptr );

        uint32_t size = 0;
        Buffer   buffer;
        Buffer   idsBuffer;
        float    additionalRadius = 0.f;
    };

    class Program;
    class ProgramManager;
    class System;
    class SphereHandler : public GeometryHandler
    {
      public:
         SphereHandler() = default;
         SphereHandler( ProgramManager & manager );
        ~SphereHandler() override;

                        SphereHandler( const SphereHandler & ) = delete;
        SphereHandler & operator=( const SphereHandler & )     = delete;

                        SphereHandler( SphereHandler && ) noexcept;
        SphereHandler & operator=( SphereHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_program = nullptr;
        UniformBuffer m_uniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_SPHERE_GEOMETRY_HPP

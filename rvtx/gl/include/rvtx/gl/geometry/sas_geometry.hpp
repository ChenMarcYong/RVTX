#ifndef RVTX_GL_GEOMETRY_SAS_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_SAS_GEOMETRY_HPP

#include <optional>

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"

namespace rvtx::gl
{
    struct SasHolder
    {
         SasHolder() = default;
        ~SasHolder() = default;

                    SasHolder( const SasHolder & ) = delete;
        SasHolder & operator=( const SasHolder & ) = delete;

                    SasHolder( SasHolder && ) noexcept = default;
        SasHolder & operator=( SasHolder && ) noexcept = default;

        static SasHolder getMolecule( const Molecule &    molecule,
                                      const MoleculeIDs * moleculeIds      = nullptr,
                                      const float         additionalRadius = 1.4f );
        static SasHolder getNonResident( const Molecule &    molecule,
                                         const MoleculeIDs * moleculeIds      = nullptr,
                                         const float         additionalRadius = 1.4f );
        static SasHolder getSystem( const Molecule &    molecule,
                                    const MoleculeIDs * moleculeIds      = nullptr,
                                    const float         additionalRadius = 1.4f );

        uint32_t size = 0;
        Buffer   buffer;
        Buffer   idsBuffer;
        float    additionalRadius = 0.f;
    };

    class Program;
    class ProgramManager;
    class System;
    class SasHandler : public GeometryHandler
    {
      public:
         SasHandler() = default;
         SasHandler( ProgramManager & manager );
        ~SasHandler() override;

                     SasHandler( const SasHandler & ) = delete;
        SasHandler & operator=( const SasHandler & )  = delete;

                     SasHandler( SasHandler && ) noexcept;
        SasHandler & operator=( SasHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_program = nullptr;
        UniformBuffer m_uniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_SAS_GEOMETRY_HPP

#ifndef RVTX_GL_GEOMETRY_STICKS_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_STICKS_GEOMETRY_HPP

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"

namespace rvtx::gl
{
    struct SticksHolder
    {
         SticksHolder() = default;
        ~SticksHolder() = default;

                       SticksHolder( const SticksHolder & ) = delete;
        SticksHolder & operator=( const SticksHolder & )    = delete;

                       SticksHolder( SticksHolder && ) noexcept = default;
        SticksHolder & operator=( SticksHolder && ) noexcept    = default;

        static SticksHolder getMolecule( const Molecule &    molecule,
                                         const MoleculeIDs * moleculeIds = nullptr,
                                         const float         radius      = 0.15f );
        static SticksHolder getNonResident( const Molecule &    molecule,
                                            const MoleculeIDs * moleculeIds = nullptr,
                                            const float         radius      = 0.15f );
        static SticksHolder getSystem( const Molecule &    molecule,
                                       const MoleculeIDs * moleculeIds = nullptr,
                                       const float         radius      = 0.15f );

        uint32_t sphereSize = 0;
        Buffer   sphereBuffer;
        Buffer   sphereIdsBuffer;
        float    sphereAdditionalRadius = 0.f;

        uint32_t cylinderSize = 0;
        Buffer   cylinderBuffer;
        Buffer   cylinderIdsBuffer;
        float    cylinderRadius = 0.15f;
    };

    class Program;
    class ProgramManager;
    class System;
    class SticksHandler : public GeometryHandler
    {
      public:
         SticksHandler() = default;
         SticksHandler( ProgramManager & manager );
        ~SticksHandler() override;

                        SticksHandler( const SticksHandler & ) = delete;
        SticksHandler & operator=( const SticksHandler & )     = delete;

                        SticksHandler( SticksHandler && ) noexcept;
        SticksHandler & operator=( SticksHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_sphereProgram   = nullptr;
        Program *     m_cylinderProgram = nullptr;
        UniformBuffer m_uniforms {};
        UniformBuffer m_cylinderUniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_STICKS_GEOMETRY_HPP

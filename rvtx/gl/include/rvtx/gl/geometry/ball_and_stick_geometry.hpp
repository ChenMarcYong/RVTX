#ifndef RVTX_GL_GEOMETRY_BALL_AND_STICK_GEOMETRY_HPP
#define RVTX_GL_GEOMETRY_BALL_AND_STICK_GEOMETRY_HPP

#include "rvtx/gl/core/fwd.hpp"
#include "rvtx/gl/geometry/handler.hpp"
#include "rvtx/gl/utils/buffer.hpp"
#include "rvtx/gl/utils/uniform.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/system/molecule_ids.hpp"

namespace rvtx::gl
{
    struct BallAndStickHolder
    {
         BallAndStickHolder() = default;
        ~BallAndStickHolder() = default;

                             BallAndStickHolder( const BallAndStickHolder & ) = delete;
        BallAndStickHolder & operator=( const BallAndStickHolder & )          = delete;

                             BallAndStickHolder( BallAndStickHolder && ) noexcept = default;
        BallAndStickHolder & operator=( BallAndStickHolder && ) noexcept          = default;

        static BallAndStickHolder getMolecule( const Molecule &    molecule,
                                               const MoleculeIDs * moleculeIds = nullptr,
                                               const float         radius      = 0.4f );
        static BallAndStickHolder getNonResident( const Molecule &    molecule,
                                                  const MoleculeIDs * moleculeIds = nullptr,
                                                  const float         radius      = 0.4f );
        static BallAndStickHolder getSystem( const Molecule &    molecule,
                                             const MoleculeIDs * moleculeIds = nullptr,
                                             const float         radius      = 0.4f );

        uint32_t sphereSize = 0;
        Buffer   sphereBuffer;
        Buffer   sphereIdsBuffer;
        float    sphereAdditionalRadius = 0.f;

        uint32_t cylinderSize = 0;
        Buffer   cylinderBuffer;
        Buffer   cylinderIdsBuffer;
        float    cylinderRadius = 0.1f;
    };

    class Program;
    class ProgramManager;
    class System;
    class BallAndStickHandler : public GeometryHandler
    {
      public:
         BallAndStickHandler() = default;
         BallAndStickHandler( ProgramManager & manager );
        ~BallAndStickHandler() override;

                              BallAndStickHandler( const BallAndStickHandler & ) = delete;
        BallAndStickHandler & operator=( const BallAndStickHandler & )           = delete;

                              BallAndStickHandler( BallAndStickHandler && ) noexcept;
        BallAndStickHandler & operator=( BallAndStickHandler && ) noexcept;

        void render( const Camera & camera, const Scene & scene ) override;

      private:
        Program *     m_sphereProgram   = nullptr;
        Program *     m_cylinderProgram = nullptr;
        UniformBuffer m_uniforms {};
        UniformBuffer m_cylinderUniforms {};

        GLuint m_vao = GL_INVALID_VALUE;
    };
} // namespace rvtx::gl

#endif // RVTX_GL_GEOMETRY_BALL_AND_STICK_GEOMETRY_HPP

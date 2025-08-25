#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_HPP

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/data.cuh"
#include "rvtx/optix/geometry.cuh"
#include "rvtx/optix/material/material.cuh"
#include "rvtx/optix/program.cuh"

namespace rvtx::optix
{
    class BallAndStickGeometry : public BaseGeometry
    {
      public:
        BallAndStickGeometry( const Context &  optixContext,
                              const Molecule & molecule,
                              float            bondRadius   = 0.15f,
                              float            sphereRadius = 0.4f );

        void                          build() override;
        uint32_t                      getGeometryNb() const override;
        std::vector<GeometryHitGroup> getGeometryData() const;

        inline Material &       getCustomMaterial();
        inline void             createCustomMaterial( const MaterialParameters & materialParameters );
        inline const Material & getCustomMaterial() const;

        inline ColorMode   getColorMode() const;
        inline ColorMode & getColorMode();
        inline void        setColorMode( ColorMode colorMode );

      private:
        const Molecule * m_molecule = nullptr;
        float            m_bondRadius;
        float            m_sphereRadius;

        Material  m_customMaterial;
        ColorMode m_colorMode = ColorMode::Atom;

        cuda::DeviceBuffer m_dGasOutputBuffer;
        cuda::DeviceBuffer m_dData;
        cuda::DeviceBuffer m_dMaterials;
        cuda::DeviceBuffer m_dSpheres;
        cuda::DeviceBuffer m_dBonds;
    };
} // namespace rvtx::optix

#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.inl"

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_GEOMETRY_HPP

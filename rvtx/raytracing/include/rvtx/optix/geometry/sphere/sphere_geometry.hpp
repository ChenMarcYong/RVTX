#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_HPP

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/data.cuh"
#include "rvtx/optix/geometry.cuh"
#include "rvtx/optix/program.cuh"

namespace rvtx::optix
{
    class SphereGeometry : public BaseGeometry
    {
      public:
        SphereGeometry( const Context & optixContext, const Molecule & molecule );

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

        Material  m_customMaterial;
        ColorMode m_colorMode = ColorMode::Atom;

        cuda::DeviceBuffer m_dGasOutputBuffer;
        cuda::DeviceBuffer m_data;
        cuda::DeviceBuffer m_materials;
    };
} // namespace rvtx::optix

#include "rvtx/optix/geometry/sphere/sphere_geometry.inl"

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_GEOMETRY_HPP

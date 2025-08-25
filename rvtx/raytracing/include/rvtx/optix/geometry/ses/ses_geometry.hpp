#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_HPP

#include <bcs/sesdf/sesdf.hpp>

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/geometry.cuh"
#include "rvtx/optix/geometry/multi_geometry_pipeline.hpp"
#include "rvtx/optix/program.cuh"

namespace rvtx::optix
{
    class SesGeometry : public BaseGeometry
    {
      public:
        SesGeometry( const Context & optixContext, const Molecule & molecule, float probeRadius = 1.4f );

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
        const Molecule *      m_molecule = nullptr;
        bcs::Aabb             m_aabb;
        bcs::Sesdf            m_sesdf;
        bcs::sesdf::SesdfData m_data;

        Material  m_customMaterial;
        ColorMode m_colorMode = ColorMode::Atom;

        std::size_t        m_primitiveNb = 0;
        cuda::DeviceBuffer m_dGasOutputBuffer;

        cuda::DeviceBuffer m_materials;
        cuda::DeviceBuffer m_circlesData;  // float4 * 4 * circleNb
        cuda::DeviceBuffer m_segmentsData; // float4 * 4 * segmentNb
        cuda::DeviceBuffer m_sesData;      // SesdfHitGroupData
    };
} // namespace rvtx::optix

#include "rvtx/optix/geometry/ses/ses_geometry.inl"

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_GEOMETRY_HPP

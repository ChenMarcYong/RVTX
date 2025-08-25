#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_HANDLER_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_HANDLER_HPP

#include "rvtx/optix/geometry/multi_geometry_pipeline.hpp"

namespace rvtx
{
    class Scene;
}

namespace rvtx::optix
{
    class SphereHandler : public GeometryHandler
    {
      public:
        SphereHandler( const Pipeline & pipeline, Scene & scene );

        std::vector<GeometryHitGroupRecord> getRecords() override;
        std::vector<OptixInstance>          getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset ) override;
        uint32_t                            getHitGroupRecordNb() const override;
        std::vector<const HitGroup *>       getHitGroups() const override;

      private:
        Scene * m_scene;

        Module          m_sphereModule;
        BuiltinISModule m_sphereISModule;
        HitGroup        m_sphereHitGroup;
    };
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_SPHERE_SPHERE_HANDLER_HPP

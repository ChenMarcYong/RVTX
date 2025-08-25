#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_HANDLER_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_SES_SES_HANDLER_HPP

#include "rvtx/optix/geometry/multi_geometry_pipeline.hpp"

namespace rvtx
{
    class Scene;
}

namespace rvtx::optix
{
    class SesHandler : public GeometryHandler
    {
      public:
        SesHandler( const Pipeline & pipeline, Scene & scene );

        std::vector<GeometryHitGroupRecord> getRecords() override;
        std::vector<OptixInstance>          getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset ) override;
        uint32_t                            getHitGroupRecordNb() const override;
        std::vector<const HitGroup *>       getHitGroups() const override;

      private:
        Scene * m_scene;

        Module   m_convexModule;
        HitGroup m_convexHitGroup;

        Module   m_concaveModule;
        HitGroup m_concaveHitGroup;

        Module   m_circleModule;
        HitGroup m_circleHitGroup;

        Module   m_segmentModule;
        HitGroup m_segmentHitGroup;

        constexpr static uint32_t PrimitiveCount = 4;
    };
} // namespace rvtx::optix

#endif // PT_RAYTRACING_SES_SESGEOMETRYHANDLER_HPP

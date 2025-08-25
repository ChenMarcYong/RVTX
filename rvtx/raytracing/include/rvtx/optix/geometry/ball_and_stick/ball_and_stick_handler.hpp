#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_HANDLER_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_HANDLER_HPP

#include "rvtx/optix/geometry/multi_geometry_pipeline.hpp"

namespace rvtx
{
    class Scene;
}

namespace rvtx::optix
{
    class BallAndStickHandler : public GeometryHandler
    {
      public:
        BallAndStickHandler( const Pipeline & pipeline, Scene & scene );

        std::vector<GeometryHitGroupRecord> getRecords() override;
        std::vector<OptixInstance>          getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset ) override;
        uint32_t                            getHitGroupRecordNb() const override;
        std::vector<const HitGroup *>       getHitGroups() const override;

      private:
        Scene * m_scene;

        Module   m_basModule;
        HitGroup m_sphereHitGroup;
        HitGroup m_bondHitGroup;

        constexpr static uint32_t PrimitiveCount = 2;
    };
} // namespace rvtx::optix

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_BALL_AND_STICK_BALL_AND_STICK_HANDLER_HPP

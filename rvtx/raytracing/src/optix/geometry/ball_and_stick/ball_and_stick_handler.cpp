#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_handler.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx::optix
{
    BallAndStickHandler::BallAndStickHandler( const Pipeline & pipeline, Scene & scene ) :
        GeometryHandler( pipeline.getContext() ), m_scene( &scene )
    {
        m_basModule = Module( *m_context, "ptx/ball_and_stick.ptx" );
        m_basModule.compile( pipeline );

        m_sphereHitGroup.setIntersection( m_basModule, "__intersection__sphere" );
        m_sphereHitGroup.setClosestHit( m_basModule, "__closesthit__sphere" );
        m_sphereHitGroup.create( *m_context );

        m_bondHitGroup.setIntersection( m_basModule, "__intersection__bond" );
        m_bondHitGroup.setClosestHit( m_basModule, "__closesthit__bond" );
        m_bondHitGroup.create( *m_context );
    }

    std::vector<GeometryHitGroupRecord> BallAndStickHandler::getRecords()
    {
        const auto                          sesGeometries = m_scene->registry.view<BallAndStickGeometry, Transform>();
        std::vector<GeometryHitGroupRecord> geometries {};

        for ( auto entity : sesGeometries )
        {
            auto & geometry = m_scene->registry.get<BallAndStickGeometry>( entity );
            geometry.build();

            std::vector<GeometryHitGroup> geometryData = geometry.getGeometryData();
            geometries.emplace_back();
            auto & sphere = geometries.back();
            m_sphereHitGroup.setSbtRecord( sphere );
            sphere.data = geometryData[ 0 ];

            geometries.emplace_back();
            auto & bond = geometries.back();
            m_bondHitGroup.setSbtRecord( bond );
            bond.data = geometryData[ 1 ];
        }

        return geometries;
    }

    std::vector<OptixInstance> BallAndStickHandler::getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset )
    {
        const auto                 sesGeometries = m_scene->registry.view<BallAndStickGeometry, Transform>();
        std::vector<OptixInstance> instances {};
        for ( const auto entity : sesGeometries )
        {
            const auto & geometry = m_scene->registry.get<BallAndStickGeometry>( entity );

            instances.emplace_back();
            OptixInstance & instance   = instances.back();
            instance.instanceId        = instanceOffset;
            instance.sbtOffset         = sbtOffset;
            instance.visibilityMask    = 255;
            instance.traversableHandle = geometry.getGASHandle();
            instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

            const Transform & transform = m_scene->registry.get<Transform>( entity );

            const glm::mat4 object = glm::transpose( transform.get() );
            std::memcpy( instance.transform, glm::value_ptr( object ), sizeof( float ) * 12 );

            instanceOffset++;
            sbtOffset += PrimitiveCount;
        }

        return instances;
    }

    uint32_t BallAndStickHandler::getHitGroupRecordNb() const
    {
        uint32_t geometryNb = 0;

        const auto sesGeometries = m_scene->registry.view<BallAndStickGeometry, Transform>();
        for ( const auto _ : sesGeometries )
            geometryNb++;

        return geometryNb * PrimitiveCount;
    }

    std::vector<const HitGroup *> BallAndStickHandler::getHitGroups() const
    {
        std::vector<const HitGroup *> hitGroups { PrimitiveCount };
        hitGroups[ 0 ] = &m_sphereHitGroup;
        hitGroups[ 1 ] = &m_bondHitGroup;

        return hitGroups;
    }

} // namespace rvtx::optix

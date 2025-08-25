#include "rvtx/optix/geometry/ses/ses_handler.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "rvtx/optix/geometry/ses/ses_geometry.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx::optix
{
    SesHandler::SesHandler( const Pipeline & pipeline, Scene & scene ) :
        GeometryHandler( pipeline.getContext() ), m_scene( &scene )
    {
        m_convexModule = Module( *m_context, "ptx/ses_convex.ptx" );
        m_convexModule.compile( pipeline );
        m_convexHitGroup.setIntersection( m_convexModule, "__intersection__convex" );
        m_convexHitGroup.setClosestHit( m_convexModule, "__closesthit__convex" );
        m_convexHitGroup.create( *m_context );

        m_concaveModule = Module( *m_context, "ptx/ses_concave.ptx" );
        m_concaveModule.compile( pipeline );
        m_concaveHitGroup.setIntersection( m_concaveModule, "__intersection__concave" );
        m_concaveHitGroup.setClosestHit( m_concaveModule, "__closesthit__concave" );
        m_concaveHitGroup.create( *m_context );

        m_circleModule = Module( *m_context, "ptx/ses_circle.ptx" );
        m_circleModule.compile( pipeline );
        m_circleHitGroup.setIntersection( m_circleModule, "__intersection__circle" );
        m_circleHitGroup.setClosestHit( m_circleModule, "__closesthit__circle" );
        m_circleHitGroup.create( *m_context );

        m_segmentModule = Module( *m_context, "ptx/ses_segment.ptx" );
        m_segmentModule.compile( pipeline );
        m_segmentHitGroup.setIntersection( m_segmentModule, "__intersection__segment" );
        m_segmentHitGroup.setClosestHit( m_segmentModule, "__closesthit__segment" );
        m_segmentHitGroup.create( *m_context );
    }

    std::vector<GeometryHitGroupRecord> SesHandler::getRecords()
    {
        const auto                          sesGeometries = m_scene->registry.view<SesGeometry, Transform>();
        std::vector<GeometryHitGroupRecord> geometries {};

        for ( auto entity : sesGeometries )
        {
            SesGeometry & geometry = m_scene->registry.get<SesGeometry>( entity );
            geometry.build();

            std::vector<GeometryHitGroup> geometryData = geometry.getGeometryData();
            geometries.emplace_back();
            auto & convex = geometries.back();
            m_convexHitGroup.setSbtRecord( convex );
            convex.data = geometryData[ 0 ];

            geometries.emplace_back();
            auto & concave = geometries.back();
            m_concaveHitGroup.setSbtRecord( concave );
            concave.data = geometryData[ 1 ];

            geometries.emplace_back();
            auto & circle = geometries.back();
            m_circleHitGroup.setSbtRecord( circle );
            circle.data = geometryData[ 2 ];

            geometries.emplace_back();
            auto & segment = geometries.back();
            m_segmentHitGroup.setSbtRecord( segment );
            segment.data = geometryData[ 3 ];
        }

        return geometries;
    }

    std::vector<OptixInstance> SesHandler::getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset )
    {
        const auto                 sesGeometries = m_scene->registry.view<SesGeometry, Transform>();
        std::vector<OptixInstance> instances {};
        for ( const auto entity : sesGeometries )
        {
            const SesGeometry & geometry = m_scene->registry.get<SesGeometry>( entity );

            instances.emplace_back();
            OptixInstance & instance   = instances.back();
            instance.instanceId        = instanceOffset;
            instance.sbtOffset         = sbtOffset;
            instance.visibilityMask    = 255;
            instance.traversableHandle = geometry.getGASHandle();
            instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
            instance.instanceId        = instanceOffset;

            const Transform & transform = m_scene->registry.get<Transform>( entity );

            // OptiX transform is row-major while glm is column-major
            const glm::mat4 object = glm::transpose( transform.get() );
            std::memcpy( instance.transform, glm::value_ptr( object ), sizeof( float ) * 12 );

            instanceOffset++;
            sbtOffset += PrimitiveCount;
        }

        return instances;
    }

    uint32_t SesHandler::getHitGroupRecordNb() const
    {
        uint32_t geometryNb = 0;

        const auto sesGeometries = m_scene->registry.view<SesGeometry, Transform>();
        for ( const auto _ : sesGeometries )
            geometryNb++;

        return geometryNb * PrimitiveCount;
    }

    std::vector<const HitGroup *> SesHandler::getHitGroups() const
    {
        std::vector<const HitGroup *> hitGroups { PrimitiveCount };
        hitGroups[ 0 ] = &m_convexHitGroup;
        hitGroups[ 1 ] = &m_concaveHitGroup;
        hitGroups[ 2 ] = &m_circleHitGroup;
        hitGroups[ 3 ] = &m_segmentHitGroup;

        return hitGroups;
    }
} // namespace rvtx::optix

#include "rvtx/optix/geometry/sphere/sphere_handler.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "rvtx/optix/geometry/sphere/sphere_geometry.hpp"
#include "rvtx/system/scene.hpp"
#include "rvtx/system/transform.hpp"

namespace rvtx::optix
{
    SphereHandler::SphereHandler( const Pipeline & pipeline, Scene & scene ) :
        GeometryHandler( pipeline.getContext() ), m_scene( &scene )
    {
        m_sphereModule = Module( *m_context, "ptx/sphere.ptx" );
        m_sphereModule.compile( pipeline );

        m_sphereISModule = BuiltinISModule( *m_context, BuiltinISModuleType::Sphere );
        m_sphereISModule.compile( pipeline );

        m_sphereHitGroup.setIntersection( m_sphereISModule, "" );
        m_sphereHitGroup.setClosestHit( m_sphereModule, "__closesthit__sphere" );
        m_sphereHitGroup.create( *m_context );
    }

    std::vector<GeometryHitGroupRecord> SphereHandler::getRecords()
    {
        const auto                          sesGeometries = m_scene->registry.view<SphereGeometry, Transform>();
        std::vector<GeometryHitGroupRecord> geometries {};

        for ( auto entity : sesGeometries )
        {
            SphereGeometry & geometry = m_scene->registry.get<SphereGeometry>( entity );
            geometry.build();

            std::vector<GeometryHitGroup> geometryData = geometry.getGeometryData();
            geometries.emplace_back();
            auto & sphere = geometries.back();
            m_sphereHitGroup.setSbtRecord( sphere );
            sphere.data = geometryData[ 0 ];
        }

        return geometries;
    }

    std::vector<OptixInstance> SphereHandler::getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset )
    {
        const auto                 sesGeometries = m_scene->registry.view<SphereGeometry, Transform>();
        std::vector<OptixInstance> instances {};
        for ( const auto entity : sesGeometries )
        {
            const SphereGeometry & geometry = m_scene->registry.get<SphereGeometry>( entity );

            instances.emplace_back();
            OptixInstance & instance   = instances.back();
            instance.instanceId        = instanceOffset;
            instance.sbtOffset         = sbtOffset;
            instance.visibilityMask    = 255;
            instance.traversableHandle = geometry.getGASHandle();
            instance.flags             = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;

            const Transform & transform = m_scene->registry.get<Transform>( entity );

            // OptiX transform is row-major while glm is column-major
            const glm::mat4 object = glm::transpose( transform.get() );
            std::memcpy( instance.transform, glm::value_ptr( object ), sizeof( float ) * 12 );

            instanceOffset++;
            sbtOffset++;
        }

        return instances;
    }

    uint32_t SphereHandler::getHitGroupRecordNb() const
    {
        uint32_t geometryNb = 0;

        const auto sesGeometries = m_scene->registry.view<SphereGeometry, Transform>();
        for ( const auto _ : sesGeometries )
            geometryNb++;

        return geometryNb;
    }

    std::vector<const HitGroup *> SphereHandler::getHitGroups() const
    {
        std::vector<const HitGroup *> hitGroups { 1 };
        hitGroups[ 0 ] = &m_sphereHitGroup;

        return hitGroups;
    }
} // namespace rvtx::optix

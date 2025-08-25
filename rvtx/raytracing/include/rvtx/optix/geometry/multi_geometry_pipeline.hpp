#ifndef RVTX_RAYTRACING_OPTIX_GEOMETRY_MULTI_GEOMETRY_PIPELINE_HPP
#define RVTX_RAYTRACING_OPTIX_GEOMETRY_MULTI_GEOMETRY_PIPELINE_HPP

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/optix/data.cuh"
#include "rvtx/optix/pipeline.cuh"
#include "rvtx/optix/program.cuh"

namespace optix
{
    struct Material;
}

namespace rvtx::optix
{
    using GeometryHitGroupRecord = Record<GeometryHitGroup>;

    using ModuleFunction = std::pair<Module *, std::string>;
    class GeometryHandler
    {
      public:
        GeometryHandler( const Context & context );
        virtual ~GeometryHandler() = default;

        virtual std::vector<GeometryHitGroupRecord> getRecords()                                                    = 0;
        virtual std::vector<OptixInstance>          getInstances( uint32_t & instanceOffset, uint32_t & sbtOffset ) = 0;
        virtual uint32_t                            getHitGroupRecordNb() const                                     = 0;
        virtual std::vector<const HitGroup *>       getHitGroups() const                                            = 0;

      protected:
        const Context * m_context;
    };

    class MultiGeometryPipeline : public Pipeline
    {
      public:
        MultiGeometryPipeline() = default;
        MultiGeometryPipeline( const Context & context );

        MultiGeometryPipeline( const MultiGeometryPipeline & )             = delete;
        MultiGeometryPipeline & operator=( const MultiGeometryPipeline & ) = delete;

        MultiGeometryPipeline( MultiGeometryPipeline && other ) noexcept;
        MultiGeometryPipeline & operator=( MultiGeometryPipeline && ) noexcept;

        ~MultiGeometryPipeline() = default;

        inline void setRayGen( Module & oModule, std::string name );
        inline void setMiss( Module & oModule, std::string name );

        template<class DerivedGeometryHandler, class... Args>
        DerivedGeometryHandler & add( Args &&... args );

        void compile() override;
        void updateGeometry();

        inline OptixTraversableHandle getHandle() const;

      private:
        ModuleFunction m_rayGenModule;
        ModuleFunction m_missModule;

        RayGen m_rayGen;
        Miss   m_missGroup;

        std::vector<std::unique_ptr<GeometryHandler>> m_handles;
        std::vector<OptixInstance>                    m_instances {};

        cuda::DeviceBuffer m_dRayGenRecord;
        cuda::DeviceBuffer m_dMissRecord;
        cuda::DeviceBuffer m_dHitGroupRecord;
        cuda::DeviceBuffer m_dIas;

        OptixTraversableHandle m_handle;

        uint32_t m_hitGroupRecordNb = 0;
    };
} // namespace rvtx::optix

#include "rvtx/optix/geometry/multi_geometry_pipeline.inl"

#endif // RVTX_RAYTRACING_OPTIX_GEOMETRY_MULTI_GEOMETRY_PIPELINE_HPP

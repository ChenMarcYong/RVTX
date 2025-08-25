#include "rvtx/optix/geometry/sphere/sphere_geometry.cuh"

#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/geometry/sphere/sphere_geometry.hpp"

namespace rvtx::optix
{
    SphereGeometry::SphereGeometry( const Context & optixContext, const Molecule & molecule ) :
        BaseGeometry( optixContext ), m_molecule( &molecule )
    {
    }

    void SphereGeometry::build()
    {
        const uint32_t primitiveNb = getGeometryNb();

        if ( m_materials.size<Material>() < m_molecule->residentAtoms.size() )
            m_materials = cuda::DeviceBuffer::Typed<Material>( m_molecule->residentAtoms.size() );

        std::vector<Material> materials {};
        materials.reserve( m_molecule->residentAtoms.size() );
        for ( std::size_t i = m_molecule->residentAtoms.start; i < m_molecule->residentAtoms.end; i++ )
        {
            materials.emplace_back( m_customMaterial );
            auto & material = materials.back();

            if ( m_colorMode == ColorMode::Atom )
            {
                const Atom & atom  = m_molecule->atoms[ i ];
                const auto   color = getAtomColor( atom );
                material.baseColor = make_float3( color.r, color.g, color.b );
            }
            else if ( m_colorMode == ColorMode::Chain )
            {
                const Atom &      atom      = m_molecule->atoms[ i ];
                const std::size_t residueId = atom.residueId;
                const std::size_t chainId   = m_molecule->residues[ residueId ].chainId;
                const Chain &     chain     = m_molecule->chains[ chainId ];

                const auto color   = getChainColor( chain );
                material.baseColor = make_float3( color.r, color.g, color.b );
            }
        }
        cuda::cudaCheck( cudaMemcpy( m_materials.get(),
                                     materials.data(),
                                     sizeof( optix::Material ) * m_molecule->residentAtoms.size(),
                                     cudaMemcpyHostToDevice ) );

        auto dSbtIndex = cuda::DeviceBuffer::Typed<uint32_t>( primitiveNb );
        cuda::cudaCheck( cudaMemset( dSbtIndex.get(), 0, dSbtIndex.size() ) );

        if ( m_data.size<float4>() < primitiveNb )
            m_data = cuda::DeviceBuffer::Typed<float4>( primitiveNb );

        cuda::cudaCheck( cudaMemcpy(
            m_data.get(), m_molecule->data.data(), sizeof( float4 ) * primitiveNb, cudaMemcpyHostToDevice ) );

        OptixBuildInput aabbInput = {};
        aabbInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

        aabbInput.sphereArray.numVertices          = primitiveNb;
        aabbInput.sphereArray.primitiveIndexOffset = 0;

        const CUdeviceptr spherePtr               = reinterpret_cast<CUdeviceptr>( m_data.get() );
        aabbInput.sphereArray.vertexBuffers       = &spherePtr;
        aabbInput.sphereArray.vertexStrideInBytes = sizeof( float4 );

        const CUdeviceptr radiusPtr               = reinterpret_cast<CUdeviceptr>( m_data.get<float>() + 3 );
        aabbInput.sphereArray.radiusBuffers       = &radiusPtr;
        aabbInput.sphereArray.radiusStrideInBytes = sizeof( float4 );

        const uint32_t aabbInputFlags                     = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; // One per SBt
        aabbInput.sphereArray.numSbtRecords               = 1;
        aabbInput.sphereArray.flags                       = &aabbInputFlags;
        aabbInput.sphereArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>( dSbtIndex.get() );
        aabbInput.sphereArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        aabbInput.sphereArray.sbtIndexOffsetStrideInBytes = 0;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;

        optixCheck( optixAccelComputeMemoryUsage(
            m_context->getOptiXContext(), &accelOptions, &aabbInput, 1, &gasBufferSizes ) );

        auto dTempBufferGas = cuda::DeviceBuffer::Typed<uint8_t>( gasBufferSizes.tempSizeInBytes );

        // non-compacted output and size of compacted GAS
        const std::size_t compactedSizeOffset = ( ( gasBufferSizes.outputSizeInBytes + 8ull - 1ull ) / 8ull ) * 8ull;
        auto dBufferTempOutputGasAndCompactedSize = cuda::DeviceBuffer::Typed<uint8_t>( compactedSizeOffset + 8 );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result
            = reinterpret_cast<CUdeviceptr>( dBufferTempOutputGasAndCompactedSize.get() + compactedSizeOffset );

        optixCheck( optixAccelBuild( m_context->getOptiXContext(),
                                     m_context->getStream(),
                                     &accelOptions,
                                     &aabbInput,
                                     1,
                                     reinterpret_cast<CUdeviceptr>( dTempBufferGas.get() ),
                                     gasBufferSizes.tempSizeInBytes,
                                     reinterpret_cast<CUdeviceptr>( dBufferTempOutputGasAndCompactedSize.get() ),
                                     gasBufferSizes.outputSizeInBytes,
                                     &m_gasHandle,
                                     &emitProperty,
                                     1 ) );

        std::size_t compactedGasSize;
        cuda::cudaCheck( cudaMemcpy(
            &compactedGasSize, (void *)emitProperty.result, sizeof( std::size_t ), cudaMemcpyDeviceToHost ) );
        if ( compactedGasSize < gasBufferSizes.outputSizeInBytes )
        {
            m_dGasOutputBuffer = cuda::DeviceBuffer::Typed<uint8_t>( compactedGasSize );

            // use handle as input and output
            optixCheck( optixAccelCompact( m_context->getOptiXContext(),
                                           m_context->getStream(),
                                           m_gasHandle,
                                           reinterpret_cast<CUdeviceptr>( m_dGasOutputBuffer.get() ),
                                           compactedGasSize,
                                           &m_gasHandle ) );
        }
        else
        {
            m_dGasOutputBuffer = std::move( dBufferTempOutputGasAndCompactedSize );
        }
    }

    uint32_t SphereGeometry::getGeometryNb() const { return static_cast<uint32_t>( m_molecule->residentAtoms.size() ); }

    std::vector<GeometryHitGroup> SphereGeometry::getGeometryData() const
    {
        GeometryHitGroup hitGroup;
        hitGroup.materials = m_materials.get<optix::Material>();
        hitGroup.userPtr   = m_data.get();

        // 1 SBT per geometry type
        return { hitGroup };
    }

} // namespace rvtx::optix

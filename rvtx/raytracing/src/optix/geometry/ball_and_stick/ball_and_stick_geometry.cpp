#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.hpp"

#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_data.hpp"
#include "rvtx/optix/geometry/ball_and_stick/ball_and_stick_geometry.cuh"

namespace rvtx::optix
{
    BallAndStickGeometry::BallAndStickGeometry( const Context &  optixContext,
                                                const Molecule & molecule,
                                                float            bondRadius,
                                                float            sphereRadius ) :
        BaseGeometry( optixContext ),
        m_molecule( &molecule ), m_bondRadius( bondRadius ), m_sphereRadius( sphereRadius )
    {
    }

    void BallAndStickGeometry::build()
    {
        const uint32_t primitiveNb = getGeometryNb();

        if ( m_dMaterials.size<Material>() < m_molecule->residentAtoms.size() )
            m_dMaterials = cuda::DeviceBuffer::Typed<Material>( m_molecule->residentAtoms.size() );

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

        cuda::cudaCheck( cudaMemcpy( m_dMaterials.get(),
                                     materials.data(),
                                     sizeof( Material ) * m_molecule->residentAtoms.size(),
                                     cudaMemcpyHostToDevice ) );

        auto dSbtIndex = cuda::DeviceBuffer::Typed<uint32_t>( primitiveNb );

        const uint32_t sphereNb = m_molecule->residentAtoms.size();
        if ( m_dSpheres.size<float4>() < sphereNb )
            m_dSpheres = cuda::DeviceBuffer::Typed<float4>( sphereNb );

        cuda::cudaCheck( cudaMemcpy(
            m_dSpheres.get(), m_molecule->data.data(), sizeof( float4 ) * sphereNb, cudaMemcpyHostToDevice ) );

        const uint32_t bondNb = m_molecule->bonds.size();
        if ( m_dBonds.size<uint2>() < bondNb )
            m_dBonds = cuda::DeviceBuffer::Typed<uint2>( bondNb );

        std::vector<uint2> bonds {};
        bonds.reserve( bondNb );
        for ( const auto & [ start, end ] : m_molecule->bonds )
            bonds.emplace_back( make_uint2( static_cast<uint32_t>( start ), static_cast<uint32_t>( end ) ) );
        cuda::cudaCheck( cudaMemcpy( m_dBonds.get(), bonds.data(), sizeof( uint2 ) * bondNb, cudaMemcpyHostToDevice ) );

        auto dAabb = cuda::DeviceBuffer::Typed<OptixAabb>( primitiveNb );
        fillBallAndStick( sphereNb,
                          m_dSpheres.get<float4>(),
                          m_sphereRadius,
                          bondNb,
                          m_dBonds.get<uint2>(),
                          m_bondRadius,
                          dAabb.get<OptixAabb>(),
                          dSbtIndex.get<uint32_t>() );

        BallAndStickHitGroupData ballAndStickHitGroupData;
        ballAndStickHitGroupData.spheres    = m_dSpheres.get<float4>();
        ballAndStickHitGroupData.sphereNb   = sphereNb;
        ballAndStickHitGroupData.bonds      = m_dBonds.get<uint2>();
        ballAndStickHitGroupData.bondNb     = bondNb;
        ballAndStickHitGroupData.bondRadius = m_bondRadius;

        m_dData = cuda::DeviceBuffer::Typed<BallAndStickHitGroupData>( 1 );
        cuda::cudaCheck( cudaMemcpy(
            m_dData.get(), &ballAndStickHitGroupData, sizeof( ballAndStickHitGroupData ), cudaMemcpyHostToDevice ) );

        OptixBuildInput aabbInput = {};
        aabbInput.type            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        const CUdeviceptr aabbPtr                           = reinterpret_cast<CUdeviceptr>( dAabb.get() );
        aabbInput.customPrimitiveArray.aabbBuffers          = &aabbPtr;
        aabbInput.customPrimitiveArray.numPrimitives        = primitiveNb;
        aabbInput.customPrimitiveArray.primitiveIndexOffset = 0;
        aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>( dSbtIndex.get() );

        constexpr uint32_t PrimitiveCount = 2;
        const auto         aabbInputFlags
            = std::vector<uint32_t>( PrimitiveCount, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ); // One per SBt
        aabbInput.customPrimitiveArray.numSbtRecords               = aabbInputFlags.size();
        aabbInput.customPrimitiveArray.flags                       = aabbInputFlags.data();
        aabbInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        aabbInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

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

    uint32_t BallAndStickGeometry::getGeometryNb() const
    {
        return static_cast<uint32_t>( m_molecule->residentAtoms.size() + m_molecule->bonds.size() );
    }

    std::vector<GeometryHitGroup> BallAndStickGeometry::getGeometryData() const
    {
        GeometryHitGroup hitGroup;
        hitGroup.materials = m_dMaterials.get<Material>();
        hitGroup.userPtr   = m_dData.get();

        // 1 SBT per geometry type
        return { hitGroup, hitGroup };
    }
} // namespace rvtx::optix

#include "rvtx/optix/geometry/ses/ses_geometry.cuh"

#include "rvtx/cuda/buffer.cuh"
#include "rvtx/molecule/color.hpp"
#include "rvtx/molecule/molecule.hpp"
#include "rvtx/optix/geometry/ses/ses_data.hpp"
#include "rvtx/optix/geometry/ses/ses_geometry.hpp"

namespace rvtx::optix
{
    SesGeometry::SesGeometry( const Context & optixContext, const Molecule & molecule, float probeRadius ) :
        BaseGeometry( optixContext ), m_molecule( &molecule ),
        m_aabb( bcs::getAabb( { molecule.data.data(), molecule.data.size() } ) ),
        m_sesdf( { molecule.data.data(), molecule.data.size() }, m_aabb, probeRadius, true, false ), m_data( m_sesdf.getData() )
    {
    }

    void SesGeometry::build()
    {
        const uint32_t primitiveNb = getGeometryNb();

        if ( m_materials.size<Material>() < m_molecule->data.size() )
            m_materials = cuda::DeviceBuffer::Typed<Material>( m_molecule->data.size() );

        std::vector<Material> materials {};
        materials.reserve( m_molecule->data.size() );

        const std::vector<uint32_t> atomIndices = m_sesdf.getAtomIndices();
        for ( std::size_t i : atomIndices )
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
                                     sizeof( Material ) * m_molecule->data.size(),
                                     cudaMemcpyHostToDevice ) );

        auto dAabb     = cuda::DeviceBuffer::Typed<OptixAabb>( primitiveNb );
        auto dSbtIndex = cuda::DeviceBuffer::Typed<uint32_t>( primitiveNb );

        if ( m_circlesData.size<CircleBoundingGeometry>() < m_data.circlePatchNb )
            m_circlesData = cuda::DeviceBuffer::Typed<CircleBoundingGeometry>( m_data.circlePatchNb );
        CircleBoundingGeometry * dCirclesData = m_circlesData.get<CircleBoundingGeometry>();

        if ( m_segmentsData.size<CircleBoundingGeometry>() < m_data.segmentPatchNb )
            m_segmentsData = cuda::DeviceBuffer::Typed<CircleBoundingGeometry>( m_data.segmentPatchNb );
        CircleBoundingGeometry * dSegmentsData = m_segmentsData.get<CircleBoundingGeometry>();

        fillPatches( m_data, dAabb.get<OptixAabb>(), dSbtIndex.get<uint32_t>(), dCirclesData, dSegmentsData );

        SesdfHitGroupData sesdfHitGroupData;
        sesdfHitGroupData.circlesBb  = m_circlesData.get<CircleBoundingGeometry>();
        sesdfHitGroupData.segmentsBb = m_segmentsData.get<CircleBoundingGeometry>();
        sesdfHitGroupData.data       = m_data;

        m_sesData = cuda::DeviceBuffer::Typed<SesdfHitGroupData>( 1 );
        cuda::cudaCheck(
            cudaMemcpy( m_sesData.get(), &sesdfHitGroupData, sizeof( SesdfHitGroupData ), cudaMemcpyHostToDevice ) );

        OptixBuildInput aabbInput = {};
        aabbInput.type            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        const CUdeviceptr aabbPtr                           = reinterpret_cast<CUdeviceptr>( dAabb.get() );
        aabbInput.customPrimitiveArray.aabbBuffers          = &aabbPtr;
        aabbInput.customPrimitiveArray.numPrimitives        = primitiveNb;
        aabbInput.customPrimitiveArray.primitiveIndexOffset = 0;
        aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>( dSbtIndex.get() );

        constexpr uint32_t PrimitiveTypes = 4;
        const auto         aabbInputFlags
            = std::vector<uint32_t>( PrimitiveTypes, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ); // One per SBt

        aabbInput.customPrimitiveArray.numSbtRecords               = PrimitiveTypes;
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

    uint32_t SesGeometry::getGeometryNb() const
    {
        const uint32_t segmentNb = m_data.segmentPatchNb;
        const uint32_t probeNb   = m_data.concavePatchNb;
        const uint32_t convexNb  = m_data.convexPatchNb;
        const uint32_t circleNb  = m_data.circlePatchNb;
        return segmentNb + probeNb + convexNb + circleNb;
    }

    std::vector<GeometryHitGroup> SesGeometry::getGeometryData() const
    {
        GeometryHitGroup hitGroup;
        hitGroup.materials = m_materials.get<Material>();
        hitGroup.userPtr   = m_sesData.get();

        // 1 SBT per geometry type
        return { hitGroup, hitGroup, hitGroup, hitGroup };
    }
} // namespace rvtx::optix

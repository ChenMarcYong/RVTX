#include "rvtx/mesh/loader.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "plf_nanotimer.h"
#include "rvtx/core/color.hpp"
#include "rvtx/core/logger.hpp"

namespace rvtx
{
    void loadMesh( fastgltf::Asset & asset, fastgltf::Mesh & inMesh, Mesh & outMesh )
    {
        for ( auto it = inMesh.primitives.begin(); it != inMesh.primitives.end(); ++it )
        {
            if ( it->type != fastgltf::PrimitiveType::Triangles )
                continue;

            const uint32_t nextIndex = outMesh.vertices.size();

            fastgltf::Primitive & primitive = *it;

            if ( primitive.indicesAccessor.has_value() )
            {
                auto & accessor = asset.accessors[ primitive.indicesAccessor.value() ];

                outMesh.indices.reserve( nextIndex + accessor.count );
                fastgltf::iterateAccessor<uint32_t>( asset,
                                                     accessor,
                                                     [ &outMesh, nextIndex ]( uint32_t index )
                                                     { outMesh.indices.emplace_back( nextIndex + index ); } );
            }

            auto * positionIt = it->findAttribute( "POSITION" );
            if ( positionIt == it->attributes.end() )
            {
                logger::error( "No position attribute found in mesh" );
                continue;
            }

            auto & positionAccessor = asset.accessors[ positionIt->second ];

            outMesh.vertices.resize( outMesh.vertices.size() + positionAccessor.count );

            if ( positionAccessor.bufferViewIndex.has_value() )
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    asset,
                    positionAccessor,
                    [ &outMesh, nextIndex ]( glm::vec3 pos, std::size_t idx )
                    {
                        outMesh.vertices[ nextIndex + idx ].position = { pos, 1.f };
                        outMesh.aabb.update( pos );
                    } );
            }

            auto * normalIt = it->findAttribute( "NORMAL" );
            if ( normalIt != it->attributes.end() )
            {
                auto & normalAccessor = asset.accessors[ normalIt->second ];

                if ( normalAccessor.bufferViewIndex.has_value() )
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(
                        asset,
                        normalAccessor,
                        [ &outMesh, nextIndex ]( glm::vec3 normal, std::size_t idx ) {
                            outMesh.vertices[ nextIndex + idx ].normal = { normal, 1.f };
                        } );
                }
            }
            else
            {
                logger::error( "No normal attribute found in mesh : TODO compute them!" );
                continue;
            }

            glm::vec4 baseColor { -1.f };
            if ( it->materialIndex.has_value() )
            {
                auto & material = asset.materials[ it->materialIndex.value() ];

                baseColor = glm::vec4 { material.pbrData.baseColorFactor[ 0 ],
                                        material.pbrData.baseColorFactor[ 1 ],
                                        material.pbrData.baseColorFactor[ 2 ],
                                        material.pbrData.baseColorFactor[ 3 ] };
            }

            auto * colorIt = it->findAttribute( "COLOR_0" );
            if ( colorIt != it->attributes.end() )
            {
                auto & colorAccessor = asset.accessors[ colorIt->second ];

                if ( colorAccessor.bufferViewIndex.has_value() )
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(
                        asset,
                        colorAccessor,
                        [ &outMesh, nextIndex ]( glm::vec4 color, std::size_t idx )
                        { outMesh.vertices[ nextIndex + idx ].color = color; } );
                }
            }
            else
            {
                if ( glm::all( glm::equal( baseColor, glm::vec4 { -1.f } ) ) )
                    baseColor = { Color::Beige, 1.f };

                for ( uint32_t i = nextIndex; i < outMesh.vertices.size(); i++ )
                    outMesh.vertices[ i ].color = baseColor;
            }
        }
    }

    Mesh loadMesh( const std::filesystem::path & path )
    {
        plf::nanotimer timer;
        timer.start();

        Mesh mesh;

        if ( !std::filesystem::exists( path ) )
        {
            logger::error( "Mesh file '{}' does not exist", path.string() );
            return Mesh();
        }

        fastgltf::Parser parser;

        constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::LoadGLBBuffers
                                     | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages
                                     | fastgltf::Options::GenerateMeshIndices;

        fastgltf::GltfDataBuffer data;
        data.loadFromFile( path );

        auto gltfType = fastgltf::determineGltfFileType( &data );

        auto asset = parser.loadGltf( &data, path.parent_path(), gltfOptions );
        if ( asset.error() != fastgltf::Error::None )
        {
            logger::error(
                "Failed to load glTF file '{}' : {}", path.string(), fastgltf::getErrorMessage( asset.error() ) );
            return Mesh();
        }

        for ( auto & inMesh : asset->meshes )
            loadMesh( asset.get(), inMesh, mesh );

        logger::info( "Loaded mesh '{}' with {} vertices and {} faces in {:.2f}{}",
                      path.string(),
                      mesh.vertices.size(),
                      mesh.indices.size() / 3,
                      timer.get_elapsed_ms() > 1000.f ? timer.get_elapsed_ms() / 1000.f : timer.get_elapsed_ms(),
                      timer.get_elapsed_ms() > 1000.f ? "s" : "ms" );

        return mesh;
    }
} // namespace rvtx
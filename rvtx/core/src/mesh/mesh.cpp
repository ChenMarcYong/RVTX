#include "rvtx/mesh/mesh.hpp"

#include <array>
#include <ranges>
#include <stack>
#include <unordered_set>

#include <BS_thread_pool.hpp>
#include <glm/gtx/hash.hpp>
#include <plf_nanotimer.h>

#include "rvtx/core/color.hpp"

namespace rvtx
{
    std::vector<glm::vec3> Mesh::getPositions() const
    {
        std::vector<glm::vec3> positions;
        positions.reserve( vertices.size() );

        for ( const auto & vertex : vertices )
            positions.emplace_back( vertex.position );

        return positions;
    }
    std::vector<glm::vec3> Mesh::getPositions( const glm::mat4 & transform ) const
    {
        std::vector<glm::vec3> positions;
        positions.reserve( vertices.size() );

        for ( const auto & vertex : vertices )
            positions.emplace_back( transform * vertex.position );

        return positions;
    }

    std::vector<glm::vec3> Mesh::getNormals() const
    {
        std::vector<glm::vec3> normals;
        normals.reserve( vertices.size() );

        for ( const auto & vertex : vertices )
            normals.emplace_back( vertex.normal );

        return normals;
    }

    std::vector<glm::vec4> Mesh::getColors() const
    {
        std::vector<glm::vec4> colors;
        colors.reserve( vertices.size() );

        for ( const auto & vertex : vertices )
            colors.emplace_back( vertex.color );

        return colors;
    }
    void Mesh::setColors( const std::vector<glm::vec4> & colors )
    {
        if ( colors.size() != vertices.size() )
            LOG_ERROR( "Colors size does not match vertices size, aborting!" );

        for ( std::size_t i = 0; i < vertices.size(); i++ )
            vertices[ i ].color = colors[ i ];
    }
    void Mesh::setColors( const std::vector<glm::vec3> & colors, const float alpha )
    {
        if ( colors.size() != vertices.size() )
            LOG_ERROR( "Colors size does not match vertices size, aborting!" );

        for ( std::size_t i = 0; i < vertices.size(); i++ )
            vertices[ i ].color = { colors[ i ], alpha };
    }

    std::vector<float> Mesh::computeCharges( Molecule &  molecule,
                                             const bool  updateColors,
                                             const float probeRadius,
                                             const float distance,
                                             const bool  buildAccelerationStructure )
    {
        if ( buildAccelerationStructure && !molecule.as.isBuilt() )
            molecule.buildAccelerationStructure();

        plf::nanotimer timer;
        timer.start();

        std::vector verticesCharges = std::vector( vertices.size(), 0.f );

        float lowerRange = std::numeric_limits<float>::max();
        float upperRange = std::numeric_limits<float>::lowest();

        std::mutex writeMutex;

        BS::thread_pool pool;
        pool.detach_blocks<std::size_t>(
            0,
            vertices.size(),
            [ this, &verticesCharges, &writeMutex, &lowerRange, &upperRange, &molecule, probeRadius, distance ](
                const std::size_t start, const std::size_t end )
            {
                for ( std::size_t i = start; i < end; i++ )
                {
                    Vertex & vertex = vertices[ i ];

                    const glm::vec3 point = vertex.position + vertex.normal * probeRadius;

                    verticesCharges[ i ] = molecule.computeCharge( point, distance );

                    lowerRange = std::min( lowerRange, verticesCharges[ i ] );
                    upperRange = std::max( upperRange, verticesCharges[ i ] );
                }
            } );
        pool.wait();

        for ( std::size_t i = 0; i < indices.size(); i += 3 )
        {
            std::array<std::size_t, 3> vertexIds;
            float                      triangleCharge = 0.f;

            for ( std::size_t j = 0; j < 3; j++ )
            {
                const std::size_t current = indices[ i + j ];
                vertexIds[ j ]            = current;
                triangleCharge += verticesCharges[ current ];
            }

            triangleCharge /= 3.f;

            for ( std::size_t j = 0; j < 3; j++ )
            {
                const std::size_t current       = vertexIds[ j ];
                const float       currentCharge = ( verticesCharges[ current ] + triangleCharge ) * .5f;
                verticesCharges[ current ]      = currentCharge;
            }
        }

        if ( updateColors )
        {
            pool.detach_blocks<std::size_t>(
                0,
                vertices.size(),
                [ this, &verticesCharges, &lowerRange, &upperRange ]( const std::size_t start, const std::size_t end )
                {
                    for ( std::size_t i = start; i < end; i++ )
                    {
                        const float charge
                            = ( glm::smoothstep( lowerRange, upperRange, verticesCharges[ i ] ) - .5f ) * 2.f;

                        const float alpha = glm::clamp( glm::abs( charge ), 0.f, 1.f );
                        vertices[ i ].color
                            = glm::vec4 { glm::mix( Color::White, ( charge > 0.f ? Color::Blue : Color::Red ), alpha ),
                                          vertices[ i ].color.a };
                    }
                } );
            pool.wait();
        }

        logger::info( "Charges computed in {:.2f}{}.",
                      timer.get_elapsed_ms() > 1000.f ? timer.get_elapsed_ms() / 1000.f : timer.get_elapsed_ms(),
                      timer.get_elapsed_ms() > 1000.f ? "s" : "ms" );

        return verticesCharges;
    }

} // namespace rvtx

#include "rvtx/system/acceleration_structure.hpp"

#include <glm/gtx/component_wise.hpp>

#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    AccelerationStructure::AccelerationStructure( const Molecule & molecule, const std::size_t size )
    {
        build( molecule, size );
    }

    AccelerationStructure::AccelerationStructure( const std::vector<glm::vec3> & positions,
                                                  const Aabb &                   aabb,
                                                  const size_t                   size )
    {
        build( positions, aabb, size );
    }

    // Based on particle simulation samples
    // Ref: https://github.com/zchee/cuda-sample/blob/master/5_Simulations/particles/particles_kernel_impl.cuh#L109
    glm::ivec3 AccelerationStructure::getGridPosition( const glm::vec3 & position ) const
    {
        glm::ivec3 gridPos;
        gridPos.x = static_cast<int>( std::floor( ( position.x - m_origin.x ) / m_cellSize.x ) );
        gridPos.y = static_cast<int>( std::floor( ( position.y - m_origin.y ) / m_cellSize.y ) );
        gridPos.z = static_cast<int>( std::floor( ( position.z - m_origin.z ) / m_cellSize.z ) );
        return gridPos;
    }
    std::size_t AccelerationStructure::getGridHash( glm::ivec3 gridPosition ) const
    {
        gridPosition.x = gridPosition.x & ( m_gridSize.x - 1 ); // wrap grid, assumes size is power of 2
        gridPosition.y = gridPosition.y & ( m_gridSize.y - 1 );
        gridPosition.z = gridPosition.z & ( m_gridSize.z - 1 );
        return ( ( gridPosition.z * m_gridSize.y ) * m_gridSize.x ) + ( gridPosition.y * m_gridSize.x )
               + gridPosition.x;
    }

    void AccelerationStructure::getNear( const glm::vec3 &                             worldPos,
                                         const float                                   radius,
                                         const std::function<void( std::size_t id )> & callback ) const
    {
        const int range = glm::compMax( glm::ivec3( glm::ceil( glm::vec3( radius ) / m_cellSize ) ) );

        std::vector<std::size_t> result {};

        const glm::ivec3 gridPosition = getGridPosition( worldPos );
        const glm::ivec3 minGrid      = glm::max( gridPosition - range, glm::ivec3 {} );
        const glm::ivec3 maxGrid      = glm::min( gridPosition + range, m_gridSize - 1 );
        for ( int x = minGrid.x; x <= maxGrid.x; x++ )
        {
            for ( int y = minGrid.y; y <= maxGrid.y; y++ )
            {
                for ( int z = minGrid.z; z <= maxGrid.z; z++ )
                {
                    const glm::ivec3  current { x, y, z };
                    const std::size_t hash = getGridHash( current );

                    const std::pair<std::size_t, std::size_t> & cubeContent = m_grid[ hash ];
                    for ( std::size_t i = cubeContent.first; i < cubeContent.second; i++ )
                    {
                        callback( m_hashes[ i ].first );
                    }
                }
            }
        }
    }

    std::vector<std::size_t> AccelerationStructure::getNear( const glm::vec3 & worldPos, const float radius ) const
    {
        const int range = glm::compMax( glm::ivec3( glm::ceil( glm::vec3( radius ) / m_cellSize ) ) );

        std::vector<std::size_t> result;

        const glm::ivec3 gridPosition = getGridPosition( worldPos );
        const glm::ivec3 minGrid      = glm::max( gridPosition - range, glm::ivec3 {} );
        const glm::ivec3 maxGrid      = glm::min( gridPosition + range, m_gridSize - 1 );
        for ( int x = minGrid.x; x <= maxGrid.x; x++ )
        {
            for ( int y = minGrid.y; y <= maxGrid.y; y++ )
            {
                for ( int z = minGrid.z; z <= maxGrid.z; z++ )
                {
                    const glm::ivec3  current { x, y, z };
                    const std::size_t hash = getGridHash( current );

                    const std::pair<std::size_t, std::size_t> & cubeContent = m_grid[ hash ];
                    for ( std::size_t i = cubeContent.first; i < cubeContent.second; i++ )
                    {
                        result.emplace_back( m_hashes[ i ].first );
                    }
                }
            }
        }

        return result;
    }

    bool AccelerationStructure::isBuilt() const { return m_built; }

    void AccelerationStructure::build( const Molecule & molecule, const size_t cellCount, const float padding )
    {
        const Aabb & aabb = molecule.getAabb();

        m_origin                  = aabb.min - padding;
        const glm::vec3 worldSize = glm::abs( aabb.max + padding - m_origin );
        m_gridSize                = glm::ivec3( cellCount );
        m_grid.resize( cellCount * cellCount * cellCount );
        m_cellSize = worldSize / static_cast<float>( cellCount );
        
        buildInternal( molecule );
    }

    void AccelerationStructure::build( const Molecule & molecule, const float cellSize, const float padding )
    {
        const Aabb & aabb = molecule.getAabb();

        m_origin                  = aabb.min - padding;
        const glm::vec3 worldSize = glm::abs( aabb.max + padding - m_origin );
        m_cellSize                = glm::vec3 { cellSize };
        m_gridSize                = glm::ivec3( worldSize / cellSize + 0.5f);
        m_grid.resize( m_gridSize.x * m_gridSize.y * m_gridSize.z );

        buildInternal( molecule );
    }

    void AccelerationStructure::build( const std::vector<glm::vec3> & position,
                                       const Aabb &                   aabb,
                                       const size_t                   cellCount,
                                       const float                    padding )
    {
        m_origin                  = aabb.min - padding;
        const glm::vec3 worldSize = glm::abs( aabb.max + padding - m_origin );
        m_gridSize                = glm::ivec3( cellCount );
        m_grid.resize( cellCount * cellCount * cellCount );
        m_cellSize = worldSize / static_cast<float>( cellCount );

        buildInternal( position );
    }

    void AccelerationStructure::build( const std::vector<glm::vec3> & position,
                                       const Aabb &                   aabb,
                                       const float                    cellSize,
                                       const float                    padding )
    {
        m_origin                  = aabb.min - padding;
        const glm::vec3 worldSize = glm::abs( aabb.max + padding - m_origin );
        m_cellSize                = glm::vec3 { cellSize };
        m_gridSize                = glm::ivec3( worldSize / cellSize + 0.5f );
        m_grid.resize( m_gridSize.x * m_gridSize.y * m_gridSize.z );

        buildInternal( position );

        
    }

    void AccelerationStructure::buildInternal( const Molecule & molecule )
    {
        const std::vector<glm::vec4> & position = molecule.data;
        m_hashes.resize( position.size() );

        for ( std::size_t i = molecule.residentAtoms.start; i < molecule.residentAtoms.end; i++ )
        {
            const glm::vec3  vec3Position = { position[ i ].x, position[ i ].y, position[ i ].z };
            const glm::ivec3 gridPosition = getGridPosition( vec3Position );
            m_hashes[ i ]                 = { i, getGridHash( gridPosition ) };
        }

        std::sort( m_hashes.begin(),
                   m_hashes.end(),
                   []( const std::pair<std::size_t, std::size_t> & a, const std::pair<std::size_t, std::size_t> & b )
                   { return a.second < b.second; } );

        m_grid[ m_hashes[ 0 ].second ].first = 0;
        for ( std::size_t i = 1; i < m_hashes.size(); i++ )
        {
            const std::pair<std::size_t, std::size_t> & pred    = m_hashes[ i - 1 ];
            const std::pair<std::size_t, std::size_t> & current = m_hashes[ i ];
            if ( pred.second != current.second )
            {
                m_grid[ pred.second ].second   = i;
                m_grid[ current.second ].first = i;
            }
        }
        m_grid[ m_hashes.back().second ].second = m_hashes.size();

        m_built = true;
    }

    void AccelerationStructure::buildInternal( const std::vector<glm::vec3> & position )
    {
        m_hashes.resize( position.size() );

        for ( std::size_t i = 0; i < position.size(); i++ )
        {
            const glm::vec3  vec3Position = position[ i ];
            const glm::ivec3 gridPosition = getGridPosition( vec3Position );
            m_hashes[ i ]                 = { i, getGridHash( gridPosition ) };
        }

        std::sort( m_hashes.begin(),
                   m_hashes.end(),
                   []( const std::pair<std::size_t, std::size_t> & a, const std::pair<std::size_t, std::size_t> & b )
                   { return a.second < b.second; } );

        m_grid[ m_hashes[ 0 ].second ].first = 0;
        for ( std::size_t i = 1; i < m_hashes.size(); i++ )
        {
            const std::pair<std::size_t, std::size_t> & pred    = m_hashes[ i - 1 ];
            const std::pair<std::size_t, std::size_t> & current = m_hashes[ i ];
            if ( pred.second != current.second )
            {
                m_grid[ pred.second ].second   = i;
                m_grid[ current.second ].first = i;
            }
        }
        m_grid[ m_hashes.back().second ].second = m_hashes.size();

        m_built = true;
    }
} // namespace rvtx
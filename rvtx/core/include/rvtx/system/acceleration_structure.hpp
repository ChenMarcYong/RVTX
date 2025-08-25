#ifndef RVTX_ACCELERATED_STRUCTURE_HPP
#define RVTX_ACCELERATED_STRUCTURE_HPP

#include <functional>

#include <glm/vec3.hpp>

#include "rvtx/core/aabb.hpp"

namespace rvtx
{
    class Molecule;
    class AccelerationStructure
    {
      public:
        AccelerationStructure() = default;
        AccelerationStructure( const Molecule & molecule, const std::size_t size = 16 );
        AccelerationStructure( const std::vector<glm::vec3> & positions, const Aabb & aabb, size_t size = 16 );
        ~AccelerationStructure() = default;

        bool isBuilt() const;
        void build( const Molecule & molecule, const size_t cellCount = 16, const float padding = 0.f );
        void build( const Molecule & molecule, const float cellSize, const float padding = 0.f );
        void build( const std::vector<glm::vec3> & position,
                    const Aabb &                   aabb,
                    const size_t                   cellCount = 16,
                    const float                    padding   = 0.f );
        void build( const std::vector<glm::vec3> & position,
                    const Aabb &                   aabb,
                    const float                    cellSize,
                    const float                    padding = 0.f );

        glm::ivec3  getGridPosition( const glm::vec3 & position ) const;
        std::size_t getGridHash( glm::ivec3 gridPosition ) const;

        std::vector<std::size_t> getNear( const glm::vec3 & worldPos, float radius ) const;
        void                     getNear( const glm::vec3 &                             worldPos,
                                          float                                         radius,
                                          const std::function<void( std::size_t id )> & callback ) const;

      private:
        bool m_built = false;

        glm::vec3  m_cellSize;
        glm::vec3  m_origin;
        glm::ivec3 m_gridSize;

        std::vector<std::pair<std::size_t /* atom id */, std::size_t /* hash */>>                m_hashes;
        std::vector<std::pair<std::size_t /* start hash idx */, std::size_t /* end hash idx */>> m_grid;

        void buildInternal( const Molecule & molecule );
        void buildInternal( const std::vector<glm::vec3> & position );
    };
} // namespace rvtx

#endif // RVTX_ACCELERATED_STRUCTURE_HPP
#ifndef RVTX_MESH_MESH_HPP
#define RVTX_MESH_MESH_HPP

#include <vector>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "rvtx/core/math.hpp"
#include "rvtx/molecule/molecule.hpp"

namespace rvtx
{
    struct Mesh
    {
        struct Vertex
        {
            glm::vec4 position;
            glm::vec4 normal;
            glm::vec4 color;
        };

        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> ids { 1 };

        Aabb aabb;

        std::vector<glm::vec3> getPositions() const;
        std::vector<glm::vec3> getPositions( const glm::mat4 & transform ) const;
        std::vector<glm::vec3> getNormals() const;
        std::vector<glm::vec4> getColors() const;

        void setColors( const std::vector<glm::vec4> & colors );
        void setColors( const std::vector<glm::vec3> & colors, const float alpha = 1.f );

        std::vector<float> computeCharges( Molecule &  molecule,
                                           const bool  updateColors               = true,
                                           const float probeRadius                = 1.4f,
                                           const float distance                   = 5.f,
                                           const bool  buildAccelerationStructure = true );
    };
} // namespace rvtx

#endif // RVTX_MESH_MESH_HPP

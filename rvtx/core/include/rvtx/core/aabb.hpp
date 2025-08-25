#ifndef RVTX_CORE_AABB_HPP
#define RVTX_CORE_AABB_HPP

#include <glm/vec3.hpp>
#include <rvtx/system/transform.hpp>

namespace rvtx
{
    struct Aabb
    {
        Aabb( const glm::vec3 & min = glm::vec3 { std::numeric_limits<float>::max() },
              const glm::vec3 & max = glm::vec3 { std::numeric_limits<float>::lowest() } );
        Aabb( const float min, const float max );

        glm::vec3 min;
        glm::vec3 max;

        inline static Aabb base();
        inline void        update( const glm::vec3 & p );
        inline void        update( const glm::vec4 & s );
        inline void        update( const Aabb & aabb );
        inline void        update( const std::vector<glm::vec3> & points );
        inline void        update( const std::vector<glm::vec4> & spheres );
        inline void        grow( const glm::vec3 size );
        inline void        grow( const float size );
        inline glm::vec3   getCentroid() const;
        inline float       getRadius() const;
        inline bool        isInvalid() const;
        inline void        setMin( const glm::vec3 & newMin );
        inline void        setMax( const glm::vec3 & newMax );

        Transform * attachedTransform = nullptr;

        void attachTransform( Transform * transform );

        inline bool      hasAttachedTransform() const;
        inline glm::vec3 getTCentroid() const;
        inline glm::vec3 getTMin() const;
        inline glm::vec3 getTMax() const;
    };
    using AABB = Aabb;
} // namespace rvtx

#include "rvtx/core/aabb.inl"

#endif // RVTX_CORE_COLOR_HPP
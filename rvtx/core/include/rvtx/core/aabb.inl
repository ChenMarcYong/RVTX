#include "rvtx/core/aabb.hpp"

namespace rvtx
{
    inline Aabb::Aabb( const glm::vec3 & min, const glm::vec3 & max ) : min( min ), max( max ) {}

    inline Aabb::Aabb( const float min, const float max ) : Aabb( glm::vec3 { min }, glm::vec3 { max } ) {}

    Aabb Aabb::base() { return {}; }

    inline void Aabb::setMin( const glm::vec3 & newMin ) { min = newMin; }
    inline void Aabb::setMax( const glm::vec3 & newMax ) { max = newMax; }

    void Aabb::update( const glm::vec3 & p )
    {
        min = glm::min( min, p );
        max = glm::max( max, p );
    }

    inline void Aabb::update( const glm::vec4 & s )
    {
        const glm::vec3 p { s };
        update( p + s.w );
        update( p - s.w );
    }

    inline void Aabb::update( const Aabb & aabb )
    {
        update( aabb.min );
        update( aabb.max );
    }

    inline void Aabb::update( const std::vector<glm::vec3> & points )
    {
        for ( const auto & p : points )
            update( p );
    }
    inline void Aabb::update( const std::vector<glm::vec4> & spheres )
    {
        for ( const auto & s : spheres )
            update( s );
    }

    inline void Aabb::grow( const glm::vec3 size )
    {
        min -= size;
        max += size;
    }

    inline void Aabb::grow( const float size )
    {
        min -= glm::vec3 { size };
        max += glm::vec3 { size };
    }

    inline glm::vec3 Aabb::getCentroid() const { return ( min + max ) * .5f; }
    inline float     Aabb::getRadius() const { return glm::length( max - min ) * .5f; }
    inline bool      Aabb::isInvalid() const
    {
        return min == glm::vec3 { std::numeric_limits<float>::max() }
               && max == glm::vec3 { std::numeric_limits<float>::lowest() };
    }

    inline void Aabb::attachTransform( Transform * transform ) { attachedTransform = transform; }

    inline bool Aabb::hasAttachedTransform() const { return attachedTransform != nullptr; }

    inline glm::vec3 Aabb::getTCentroid() const
    {
        return ( min + max ) * .5f + ( hasAttachedTransform() ? attachedTransform->position : glm::vec3 { 0.f } );
    }
    inline glm::vec3 Aabb::getTMin() const
    {
        return min + ( hasAttachedTransform() ? attachedTransform->position : glm::vec3 { 0.f } );
    }
    inline glm::vec3 Aabb::getTMax() const
    {
        return max + ( hasAttachedTransform() ? attachedTransform->position : glm::vec3 { 0.f } );
    }
} // namespace rvtx
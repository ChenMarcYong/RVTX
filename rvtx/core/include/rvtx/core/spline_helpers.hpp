#ifndef RVTX_CORE_SPLINE_HELPERS_HPP
#define RVTX_CORE_SPLINE_HELPERS_HPP

#include <glm/gtc/quaternion.hpp>

namespace rvtx
{
    inline glm::vec3 qlog( const glm::quat q )
    {
        const float length = glm::sqrt( q.x * q.x + q.y * q.y + q.z * q.z );

        if ( length < 1e-5f )
        {
            return { q.x, q.y, q.z };
        }

        const float halfAngle = acosf( glm::clamp( q.w, -1.0f, 1.0f ) );
        return halfAngle * ( glm::vec3( q.x, q.y, q.z ) / length );
    }

    inline glm::quat qexp( const glm::vec3 v )
    {
        const float halfAngle = glm::sqrt( v.x * v.x + v.y * v.y + v.z * v.z );

        if ( halfAngle < 1e-6f )
        {
            return glm::normalize( glm::quat { 1.0f, v.x, v.y, v.z } );
        }

        const float c = glm::cos( halfAngle );
        const float s = glm::sin( halfAngle ) / halfAngle;
        return { c, s * v.x, s * v.y, s * v.z };
    }

    inline glm::quat qinv( const glm::quat q ) { return { -q.w, q.x, q.y, q.z }; }
    inline glm::quat qinvMul( const glm::quat q, const glm::quat p ) { return qinv( q ) * p; }
    inline glm::quat qmulInv( const glm::quat q, const glm::quat p ) { return q * qinv( p ); }
    inline glm::quat qabs( const glm::quat x ) { return x.w < 0.f ? -x : x; }
    inline glm::vec3 qtoScaledAngleAxis( const glm::quat q ) { return 2.0f * qlog( q ); }
    inline glm::quat qfromScaledAngleAxis( const glm::vec3 v ) { return qexp( v / 2.0f ); }
} // namespace rvtx

#endif // RVTX_CORE_SPLINE_HELPERS_HPP

#ifndef RVTX_CORE_SPLINE_HPP
#define RVTX_CORE_SPLINE_HPP

#include "rvtx/core/spline_helpers.hpp"

namespace rvtx
{
    enum class SplineType
    {
        Linear,
        CatmullRom,
    };

    template<typename T>
    T hermite( const T v1, const T v2, const glm::vec3 vel0, const glm::vec3 vel1, const float t )
    {
        assert( "Not implemented for this type yet" );
    }

    template<>
    inline glm::quat hermite<glm::quat>( const glm::quat r0,
                                         const glm::quat r1,
                                         const glm::vec3 v0,
                                         const glm::vec3 v1,
                                         const float     t )
    {
        const float w1 = 3.f * t * t - 2.f * t * t * t;
        const float w2 = t * t * t - 2.f * t * t + t;
        const float w3 = t * t * t - t * t;

        const glm::vec3 r1_sub_r0 = qtoScaledAngleAxis( qabs( qmulInv( r1, r0 ) ) );

        return qfromScaledAngleAxis( w1 * r1_sub_r0 + w2 * v0 + w3 * v1 ) * r0;
    }

    template<typename T>
    T linear( const T & v1, const T & v2, const float t )
    {
        return v1 * ( typename T::value_type( 1 ) - t ) + v2 * t;
    }

    template<>
    inline glm::quat linear( const glm::quat & x, const glm::quat & y, const float a )
    {
        // Based on glm implementation

        glm::quat z = y;

        float cosTheta = dot( x, y );

        if ( cosTheta < 1.f )
        {
            z        = -y;
            cosTheta = -cosTheta;
        }

        if ( cosTheta > 1.f - glm::epsilon<float>() )
        {
            return glm::quat(
                glm::mix( x.w, z.w, a ), glm::mix( x.x, z.x, a ), glm::mix( x.y, z.y, a ), glm::mix( x.z, z.z, a ) );
        }
        else
        {
            float angle = acos( cosTheta );
            return ( std::sin( ( 1.f - a ) * angle ) * x + std::sin( a * angle ) * z ) / std::sin( angle );
        }
    }

    template<typename T>
    T catmullRom( const T & v1, const T & v2, const T & v3, const T & v4, const float t )
    {
        // Based on glm implementation
        typename T::value_type t2 = t * t;
        typename T::value_type t3 = t * t * t;

        typename T::value_type f1 = -t3 + typename T::value_type( 2 ) * t2 - t;
        typename T::value_type f2 =
            typename T::value_type( 3 ) * t3 - typename T::value_type( 5 ) * t2 + typename T::value_type( 2 );
        typename T::value_type f3 = typename T::value_type( -3 ) * t3 + typename T::value_type( 4 ) * t2 + t;
        typename T::value_type f4 = t3 - t2;

        return ( f1 * v1 + f2 * v2 + f3 * v3 + f4 * v4 ) / typename T::value_type( 2 );
    }

    template<>
    inline glm::quat catmullRom<glm::quat>( const glm::quat & r0,
                                            const glm::quat & r1,
                                            const glm::quat & r2,
                                            const glm::quat & r3,
                                            const float       t )
    {
        // https://theorangeduck.com/page/cubic-interpolation-quaternions

        const glm::vec3 r1_sub_r0 = qtoScaledAngleAxis( qabs( qmulInv( r1, r0 ) ) );
        const glm::vec3 r2_sub_r1 = qtoScaledAngleAxis( qabs( qmulInv( r2, r1 ) ) );
        const glm::vec3 r3_sub_r2 = qtoScaledAngleAxis( qabs( qmulInv( r3, r2 ) ) );

        const glm::vec3 v1 = ( r1_sub_r0 + r2_sub_r1 ) / 2.f;
        const glm::vec3 v2 = ( r2_sub_r1 + r3_sub_r2 ) / 2.f;

        return hermite( r1, r2, v1, v2, t );
    }
} // namespace rvtx

#endif // RVTX_CORE_SPLINE_HPP

#include <cmath>

#include "rvtx/core/random.hpp"

namespace rvtx
{
    // Tien-Tsin Wong et al., Sampling with Hammersley and Halton Points
    // Reference: https://www.cse.cuhk.edu.hk/~ttwong/papers/udpoint/udpoint.pdf
    inline std::vector<glm::vec3> sphereHalton( int n, int p2 )
    {
        auto result = std::vector<glm::vec3>( n );
        for ( int k = 0, pos = 0; k < n; k++ )
        {
            float t  = 0.f;
            int   kk = k;
            for ( float p = 0.5f; kk; p *= .5f, kk >>= 1 )
            {
                if ( kk & 1 )
                {
                    t += p;
                }
            }

            t = 2.f * t - 1.f;

            float st  = std::sqrt( 1.f - t * t );
            float phi = 0.f;
            float ip  = 1.f / p2;
            kk        = k;
            for ( float p = ip; kk; p *= ip, kk /= p2 )
            {
                int a = kk % p2;
                if ( a )
                {
                    phi += a * p;
                }
            }

            float phirad = phi * 4.f * Pi;
            result[ k ]  = { st * std::cos( phirad ), st * std::sin( phirad ), t };
        }

        return result;
    }
} // namespace rvtx

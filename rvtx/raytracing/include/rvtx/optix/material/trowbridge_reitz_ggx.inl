#include "rvtx/optix/material/trowbridge_reitz_ggx.cuh"

namespace rvtx::optix
{
    // Found: https://github.com/boksajak/brdf/blob/master/brdf.h#L710
    inline __device__ float getSmithG1GGX( float sn2, float alpha2 )
    {
        return 2.f / ( sqrtf( ( ( alpha2 * ( 1.f - sn2 ) ) + sn2 ) / sn2 ) + 1.f );
    }

    // Moving Frostbite to Physically Based Rendering by Lagarde & de Rousiers
    // Found: https://github.com/boksajak/brdf/blob/master/brdf.h#L653
    // Includes specular BRDF denominator
    inline __device__ float getSmithG2GGX( float won, float win, float alpha2 )
    {
        const float ggxv = win * sqrtf( won * won * ( 1.f - alpha2 ) + alpha2 );
        const float ggxl = won * sqrtf( win * win * ( 1.f - alpha2 ) + alpha2 );

        return 0.5f / ( ggxv + ggxl );
    }

    // Found: https://github.com/boksajak/brdf/blob/master/brdf.h#L710
    inline __device__ float getDGGX( float hn, float alpha2 )
    {
        const float b = ( ( alpha2 - 1.f ) * hn * hn + 1.f );
        return alpha2 / fmax( 1e-4f, rvtx::Pi * b * b );
    }

    // Eric Heitz, A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals,
    // Technical report 2017
    inline __device__ float3 sampleGGXVNDF( float3 V_, float alpha_x, float alpha_y, float U1, float U2 )
    {
        // stretch view
        float3 V = normalize( make_float3( alpha_x * V_.x, alpha_y * V_.y, V_.z ) );

        // orthonormal basis
        float3 T1 = ( V.z < 0.9999f ) ? normalize( cross( V, make_float3( 0, 0, 1 ) ) ) : make_float3( 1, 0, 0 );
        float3 T2 = cross( T1, V );

        // sample point with polar coordinates (r, phi)
        const float a   = 1.f / ( 1.f + V.z );
        const float r   = sqrtf( U1 );
        const float phi = ( U2 < a ) ? U2 / a * Pi : Pi + ( U2 - a ) / ( 1.f - a ) * Pi;
        const float P1  = r * cosf( phi );
        const float P2  = r * sinf( phi ) * ( ( U2 < a ) ? 1.f : V.z );

        // compute normal
        float3 N = P1 * T1 + P2 * T2 + sqrtf( fmax( 0.f, 1.f - P1 * P1 - P2 * P2 ) ) * V;

        // unstretch
        N = normalize( make_float3( alpha_x * N.x, alpha_y * N.y, fmax( 0.f, N.z ) ) );
        return N;
    }

} // namespace rvtx::optix

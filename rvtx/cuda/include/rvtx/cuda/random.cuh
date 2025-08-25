#ifndef RVTX_CUDA_RANDOM_CUH
#define RVTX_CUDA_RANDOM_CUH

#include "rvtx/core/math.hpp"
#include "rvtx/cuda/math.cuh"

namespace rvtx::cuda
{
    //
    // Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
    //
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions
    // are met:
    //  * Redistributions of source code must retain the above copyright
    //    notice, this list of conditions and the following disclaimer.
    //  * Redistributions in binary form must reproduce the above copyright
    //    notice, this list of conditions and the following disclaimer in the
    //    documentation and/or other materials provided with the distribution.
    //  * Neither the name of NVIDIA CORPORATION nor the names of its
    //    contributors may be used to endorse or promote products derived
    //    from this software without specific prior written permission.
    //
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
    // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    // PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    // CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    // OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    //
    template<unsigned int N>
    __host__ __device__ uint32_t tea( uint32_t val0, uint32_t val1 )
    {
        // https://www.csee.umbc.edu/~olano/papers/GPUTEA.pdf

        uint32_t v0 = val0;
        uint32_t v1 = val1;
        uint32_t s0 = 0;

        for ( uint32_t n = 0; n < N; n++ )
        {
            s0 += 0x9e3779b9;
            v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
            v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
        }

        return v0;
    }

    // Generate random unsigned int in [0, 2^24)
    inline __host__ __device__ unsigned int lcg( uint32_t & prev )
    {
        const unsigned int LCG_A = 1664525u;
        const unsigned int LCG_C = 1013904223u;
        prev                     = ( LCG_A * prev + LCG_C );
        return prev & 0x00FFFFFF;
    }

    struct pcg_state_setseq_64
    {                   // Internals are *Private*.
        uint64_t state; // RNG state.  All values are possible.
        uint64_t inc;   // Controls which RNG sequence (stream) is
                        // selected. Must *always* be odd.
    };
    using RandomState = pcg_state_setseq_64;

    __host__ __device__ uint32_t    pcg32_random_r( RandomState & rng );
    inline __host__ __device__ void pcg32_srandom_r( RandomState & rng, uint64_t initstate, uint64_t initseq )
    {
        rng.state = 0U;
        rng.inc   = ( initseq << 1u ) | 1u;
        pcg32_random_r( rng );
        rng.state += initstate;
        pcg32_random_r( rng );
    }

    inline __host__ __device__ uint32_t pcg32_random_r( RandomState & rng )
    {
        uint64_t oldstate   = rng.state;
        rng.state           = oldstate * 6364136223846793005ULL + rng.inc;
        uint32_t xorshifted = ( ( oldstate >> 18u ) ^ oldstate ) >> 27u;
        uint32_t rot        = oldstate >> 59u;
        return ( xorshifted >> rot ) | ( xorshifted << ( ( -rot ) & 31 ) );
    }

    inline __host__ __device__ void initSeed( RandomState & rng, uint64_t initstate, uint64_t initseq )
    {
        pcg32_srandom_r( rng, initstate, initseq );
    }

    // Generate random float in [0, 1)
    inline __host__ __device__ float rnd( RandomState & prev )
    {
        return ( (float)pcg32_random_r( prev ) / (float)0xFFFFFFFF );
    }

    // Generate random float2 in [-1, 1)
    inline __host__ __device__ float2 randomFloat2( RandomState & prev )
    {
        float2 v;
        v.x = ( rnd( prev ) - .5f ) * 2.f;
        v.y = ( rnd( prev ) - .5f ) * 2.f;
        return v;
    }
    // Generate random float3 in [-1, 1)
    inline __host__ __device__ float3 randomFloat3( RandomState & prev )
    {
        float3 v;
        v.x = ( rnd( prev ) - .5f ) * 2.f;
        v.y = ( rnd( prev ) - .5f ) * 2.f;
        v.z = ( rnd( prev ) - .5f ) * 2.f;
        return v;
    }

    // Generate random float3 in unit sphere
    inline __host__ __device__ float3 randomInSphere( RandomState & prev )
    {
        float3 vec = randomFloat3( prev );
        while ( dot( vec, vec ) >= 1.f )
            vec = randomFloat3( prev );

        return vec;
    }

    // Stratified Sampling of 2-Manifolds, Jim Arvo
    // SIGGRAPH Course Notes 2001
    // Found: https://twitter.com/keenanisalive/status/1529490555893428226?s=20&t=mxRju6YioMmlMOJ1fDVBpw
    inline __host__ __device__ float2 randomInDisk( RandomState & prev )
    {
        const float2 vec   = { rnd( prev ), rnd( prev ) };
        const float  r     = vec.x;
        const float  theta = vec.y * 2.f * Pi;
        return ::sqrtf( r ) * make_float2( cos( theta ), sin( theta ) );
    }

    // Sampling Transformations Zoo
    // Peter Shirley, Samuli Laine, David Hart, Matt Pharr, Petrik Clarberg,
    // Eric Haines, Matthias Raab, and David Cline
    // NVIDIA
    inline __host__ __device__ float3 randomInCosineWeightedHemisphere( float2 u )
    {
        // 16.6.1 COSINE-WEIGHTED HEMISPHERE ORIENTED TO THE Z-AXIS
        const float r     = u.x;
        const float theta = u.y * 2.f * Pi;
        return make_float3( ::sqrtf( r ) * make_float2( cos( theta ), sin( theta ) ), ::sqrtf( 1.f - u.x ) );
    }

    inline __host__ __device__ float halton( int base, int index )
    {
        float fBase = static_cast<float>( base );
        float r     = 0.f;
        float f     = 1.f;
        while ( index > 0 )
        {
            f /= fBase;
            r += f * static_cast<float>( index % base );
            index = floor( index / fBase );
        }
        return r;
    }
} // namespace rvtx::cuda

#endif // RVTX_CUDA_RANDOM_CUH

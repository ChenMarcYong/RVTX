#ifndef RVTX_OPTIX_MATERIAL_MATERIAL_CUH
#define RVTX_OPTIX_MATERIAL_MATERIAL_CUH

#include "rvtx/cuda/random.cuh"

namespace rvtx
{
    struct MaterialParameters;
}

namespace rvtx::optix
{
    struct Material
    {
        Material() = default;
        Material( const MaterialParameters & materialParameters );

        float3 baseColor = { 0.f, 0.f, 0.f };
        float  roughness;

        float3 emissive = { 0.f, 0.f, 0.f };
        float  metallic = 0.f;

        float3 transmittance = { 0.f, 0.f, 0.f };
        float  atDistance    = 0.f;

        float ior                  = 0.f;
        float specularTransmission = 0.f;
        float specularTint         = 0.f;
        float clearcoat            = 0.f;

        float clearcoatGloss = 0.f;
        float pad1;
        float pad2;
        float pad3;

        inline __device__ float3 evalDisneyDiffuse( float3 wo, float3 wi ) const;
        inline __device__ float3 sampleDisneyDiffuse( float3 wo, float2 u ) const;
        inline __device__ float  getPdfDisneyDiffuse( float3 wo, float3 wi ) const;

        inline __device__ float3 getDisneyFresnel( float3 wo, float3 wi, float3 h ) const;

        __device__ inline float evalSpecularReflection( float3 wo, float3 wi ) const;
        __device__ inline float getPdfSpecularReflection( float3 wo, float3 wi ) const;
        __device__ inline float evalSpecularTransmission( float3 wo, float3 wi ) const;
        __device__ inline float getPdfSpecularTransmission( float3 wo, float3 wi ) const;

        __device__ inline float getClearCoatRoughness() const;
        __device__ inline float evalClearCoat( float3 wo, float3 wi ) const;
        __device__ inline float getPDFClearCoat( float3 wo, float3 wi ) const;

        __device__ inline float3 evaluate( float3 wo, float3 wi, float t ) const;
        __device__ inline float3 sample( float3                    wo,
                                         float                     t,
                                         rvtx::cuda::RandomState & seed,
                                         float3 &                  weight,
                                         float &                   pdf ) const;
        __device__ inline float  getPdfMaterial( float3 wo, float3 wi, rvtx::cuda::RandomState & seed ) const;
    };

    inline __device__ float iorToReflectance( float ior );
} // namespace rvtx::optix

#include "rvtx/optix/material/material.inl"

#endif // RVTX_OPTIX_MATERIAL_MATERIAL_CUH

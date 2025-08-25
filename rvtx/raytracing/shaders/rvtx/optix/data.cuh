#ifndef RVTX_SHADERS_OPTIX_DATA_CUH
#define RVTX_SHADERS_OPTIX_DATA_CUH

#include <optix.h>
#include <rvtx/optix/material/material.cuh>
#include <rvtx/optix/texture.cuh>

namespace rvtx::optix
{
    struct GeometryHitGroup
    {
        const Material * materials;
        const uint8_t *  userPtr;
    };

    struct HitInfo
    {
        alignas( 16 ) Material material;
        float3 position;
        float  t;
        alignas( 16 ) float3 worldNormal;
        float3 objectNormal;
        int    hit = 0;

        __device__ __host__ bool hasHit() const { return static_cast<bool>( hit ); }
    };

    struct PathTracerCamera
    {
        float4 view[ 4 ];
        float4 projection[ 4 ];

        // Orthographic camera settings
        float distance;
        bool  isPerspective;
    };

    struct Ray
    {
        float3 origin;
        float3 direction;

        Ray() = default;

        // Perspective based on https://github.com/boksajak/referencePT/blob/master/shaders/PathTracer.hlsl#L525
        __host__ __device__ Ray( const float2 & uv, const PathTracerCamera & camera )
        {
            const float aspect      = camera.projection[ 1 ].y / camera.projection[ 0 ].x;
            const float tanHalfFovY = 1.f / camera.projection[ 1 ].y;

            const float3 worldU = uv.x * make_float3( camera.view[ 0 ] ) * tanHalfFovY * aspect;
            const float3 worldV = uv.y * make_float3( camera.view[ 1 ] ) * tanHalfFovY;

            origin = make_float3( camera.view[ 3 ] );
            if ( camera.isPerspective )
            {
                direction = normalize( worldU + worldV - make_float3( camera.view[ 2 ] ) );
            }
            else
            {
                direction = make_float3( 0.f ) - make_float3( camera.view[ 2 ] );
                origin += ( worldU + worldV ) * camera.distance; // Shift by pixel pos
            }
        }
    };

    struct PathTracerParameters
    {
        PathTracerCamera camera;
        Texture::View    environment;
        Texture::View    environmentSampling;

        uint2    viewSize;
        uint32_t subFrameId;
        uint32_t pixelWidth;

        float4 * accumulation;
        uchar4 * frame;

        float3   depthOfField; // {.x > 0: enable; .y = distance; .z = strength}
        uint32_t bounces;

        OptixTraversableHandle handle;
        float4                 background;      // {.xyz = strength; .w = alpha value when no hit a depth = 1}
        float4                 backgroundLight; // {.xyz = color; .w = strength}

        float overriddenRoughness;
        float overriddenTransmission;
        float overriddenSpecularIOR;
        float overriddenTransmissionIOR;
        float overriddenMediumWeight;
        bool  normalOnly;

        void * userPtr;
    };

} // namespace rvtx::optix

#endif // RVTX_SHADERS_OPTIX_DATA_CUH
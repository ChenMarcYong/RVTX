#include <optix.h>
#include <rvtx/cuda/math.cuh>
#include <rvtx/cuda/random.cuh>
#include <rvtx/optix/environment_sampler.cuh>
#include <rvtx/optix/material/color.cuh>

#include "data.cuh"
#include "memory.cuh"

using namespace rvtx;
using namespace rvtx::optix;
using namespace rvtx::cuda;

extern "C"
{
    __constant__ PathTracerParameters parameters;
}

// A Fast and Robust Method for Avoiding Self-Intersection, Carsten W�chter and Nikolaus Binder, NVIDIA
// Reference:
// https://github.com/Apress/ray-tracing-gems/blob/master/Ch_06_A_Fast_and_Robust_Method_for_Avoiding_Self-Intersection/offset_ray.cu
__device__ float3 offsetRay( const float3 p, const float3 n )
{
    constexpr float origin      = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale   = 256.0f;

    int3 of_i = make_int3( int_scale * n.x, int_scale * n.y, int_scale * n.z );

    float3 p_i = make_float3( __int_as_float( __float_as_int( p.x ) + ( ( p.x < 0 ) ? -of_i.x : of_i.x ) ),
                              __int_as_float( __float_as_int( p.y ) + ( ( p.y < 0 ) ? -of_i.y : of_i.y ) ),
                              __int_as_float( __float_as_int( p.z ) + ( ( p.z < 0 ) ? -of_i.z : of_i.z ) ) );

    return make_float3( fabsf( p.x ) < origin ? p.x + float_scale * n.x : p_i.x,
                        fabsf( p.y ) < origin ? p.y + float_scale * n.y : p_i.y,
                        fabsf( p.z ) < origin ? p.z + float_scale * n.z : p_i.z );
}

__device__ float2 sampleCircle( float2 u )
{
    float r     = u.x;
    float theta = u.y * 2.f * Pi;
    return ::sqrtf( r ) * make_float2( cosf( theta ), sinf( theta ) );
}

__device__ float3 sampleDisk( float height, float radius, float2 u )
{
    float2 pd = sampleCircle( u );
    return make_float3( pd.x * radius, height, pd.y * radius );
}
__device__ float iDisk( float3 ro, float3 rd, float3 c, float3 n, float r )
{
    float3 o = ro - c;
    float  t = -dot( n, o ) / dot( rd, n );
    float3 q = o + rd * t;
    return ( dot( q, q ) < r * r ) ? t : -1.f;
}

// https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling
__device__ inline float powerHeuristic( int nf, float fPdf, int ng, float gPdf )
{
    float f = float( nf ) * fPdf, g = float( ng ) * gPdf;
    return ( f * f ) / ( f * f + g * g );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3    idx             = optixGetLaunchIndex();
    const float3   backgroundColor = make_float3( parameters.background );
    const float3   backgroundLight = make_float3( parameters.backgroundLight ) * parameters.backgroundLight.w;
    const uint2    viewSize        = parameters.viewSize;
    const uint32_t subFrameId      = parameters.subFrameId;
    const uint32_t bounces         = subFrameId == 0 ? min( parameters.bounces, 16 ) : parameters.bounces;

    uint32_t pixelWidth = parameters.pixelWidth;
    if ( pixelWidth % 2 != 0 )
        pixelWidth += 1;

    const float                  fPixelWidth = static_cast<float>( pixelWidth );
    const float                  fSubframeId = static_cast<float>( subFrameId % ( pixelWidth * pixelWidth ) );
    const float                  pixelStep   = 1.f / fPixelWidth;
    const OptixTraversableHandle handle      = parameters.handle;

    RandomState seed;
    pcg32_srandom_r(
        seed, tea<3>( idx.y * viewSize.x + idx.x, subFrameId ), tea<4>( idx.y * viewSize.x + idx.x, subFrameId ) );

    float3 finalColor = make_float3( 0.f );

    const float2 jittering = make_float2( rnd( seed ), rnd( seed ) );
    const float2 alea      = make_float2( rnd( seed ), rnd( seed ) );
    const float2 subPixelId
        = make_float2( floor( fmodf( fSubframeId, fPixelWidth ) ), floor( fSubframeId / fPixelWidth ) ) + jittering;
    float2 subpixelJitter = subPixelId * pixelStep * 2.f - 1.f;

    if ( subFrameId == 0 )
        subpixelJitter = make_float2( 0.f );

    const float2 uv  = 2.f * ( make_float2( idx.x, idx.y ) + subpixelJitter ) / make_float2( viewSize ) - 1.f;
    Ray          ray = Ray( uv, parameters.camera );

    if ( parameters.depthOfField.x > 0.f )
    {
        const float3 trueCoordinate = ray.origin + ray.direction * parameters.depthOfField.y;
        const float2 randDisk       = randomInDisk( seed ) * parameters.depthOfField.z;

        ray.origin += randDisk.x * make_float3( parameters.camera.view[ 0 ] );
        ray.origin += randDisk.y * make_float3( parameters.camera.view[ 1 ] );

        ray.direction = normalize( trueCoordinate - ray.origin );
    }

    float alpha = 1.f;

    bool    lastTransmitted = false;
    HitInfo hitInfo {};
    HitInfo shadowHitInfo {};
    float3  attenuation = make_float3( 1.0f, 1.0f, 1.0f );
    uint2   radiancePrd = u64ToU32x2( &hitInfo );
    uint2   shadowPrd   = u64ToU32x2( &shadowHitInfo );
    for ( int depth = 0; depth < bounces; depth++ )
    {
        if ( isnan( ray.origin.x ) || isnan( ray.origin.y ) || isnan( ray.origin.z ) )
            break;
        if ( isnan( ray.direction.x ) || isnan( ray.direction.y ) || isnan( ray.direction.z ) )
            break;

        optixTrace( handle,
                    ray.origin,
                    ray.direction,
                    0.01f, // tmin
                    1e16f, // tmax
                    0.0f,  // rayTime
                    OptixVisibilityMask( 1 ),
                    OPTIX_RAY_FLAG_NONE,
                    0, // SBT offset
                    1, // SBT stride
                    0, // missSBTIndex
                    radiancePrd.x,
                    radiancePrd.y );

        if ( !hitInfo.hasHit() )
        {
            if ( depth == 0 )
            {
                if ( parameters.background.w == 0.f )
                    alpha = 0.;

                const float4 environment
                    = rvtx::optix::getEnvironmentColor( parameters.environment, ray.direction, 0.f );
                finalColor += attenuation * make_float3( environment.x, environment.y, environment.z );
            }
            else if ( lastTransmitted )
            {
                const float4 environment
                    = rvtx::optix::getEnvironmentColor( parameters.environment, ray.direction, 0.f );
                finalColor += attenuation * make_float3( environment.x, environment.y, environment.z );
            }
            break;
        }

        // const float3 objectNormal = normalize( hitInfo.objectNormal );
        const float3 worldNormal = normalize( hitInfo.worldNormal );
        if ( parameters.normalOnly )
        {
            finalColor = worldNormal;
            break;
        }

        const float3 wo = make_float3( 0.f ) - ray.direction;

        const float3 position = hitInfo.position;
        Material     material = hitInfo.material;

        // Emissive materials
        finalColor += attenuation * material.emissive;

        const float4 transformation = toLocalZ( worldNormal );
        const float3 nLocal         = make_float3( 0.f, 0.f, 1.f );
        const float3 woLocal        = normalize( rotate( wo, transformation ) );

        const float  inside = copysignf( 1.f, woLocal.z );
        const float3 pp     = offsetRay( position, worldNormal * inside );

        // MIS
        {
            float3 direct = make_float3( 0.f );

            // Sampling light
            {
                float        lightPdf;
                const float3 wi = sampleEnvironment( //
                    parameters.environmentSampling,
                    make_float2( rnd( seed ), rnd( seed ) ),
                    lightPdf );

                const float3 wiLocal      = normalize( rotate( wi, transformation ) );
                const float  cosTheta     = fabs( wiLocal.z );
                const bool canPassThrough = ( wiLocal.z * woLocal.z > 0.f ) || ( material.specularTransmission > 0.f );

                if ( lightPdf > 0.f && canPassThrough && !isnan( wiLocal.x ) && !isnan( wiLocal.y )
                     && !isnan( wiLocal.z ) )
                {
                    if ( isnan( wi.x ) || isnan( wi.y ) || isnan( wi.z ) )
                        break;

                    optixTrace( handle,
                                pp,
                                wi,
                                0.01f, // tmin
                                1e16f, // tmax
                                0.0f,  // rayTime
                                OptixVisibilityMask( 1 ),
                                OPTIX_RAY_FLAG_NONE,
                                0, // SBT offset
                                1, // SBT stride
                                0, // missSBTIndex
                                shadowPrd.x,
                                shadowPrd.y );
                    if ( !shadowHitInfo.hasHit() )
                    {
                        const float4 environment = rvtx::optix::getEnvironmentColor( parameters.environment, wi, 0.f );

                        const float3 intensity = make_float3( environment );
                        const float3 bsdf      = material.evaluate( woLocal, wiLocal, shadowHitInfo.t ) * cosTheta;

                        const float scatteringPdf = material.getPdfMaterial( woLocal, wiLocal, seed );
                        const float weight        = powerHeuristic( 1, lightPdf, 1, scatteringPdf );

                        direct += fminf( intensity, bsdf * intensity * weight / max( 1e-2f, lightPdf ) );
                    }
                }
            }

            // Sampling BRDF
            {
                float  scatteringPdf  = 0.f;
                float3 bsdf           = make_float3( 0.f );
                float3 wiLocal        = material.sample( woLocal, hitInfo.t, seed, bsdf, scatteringPdf );
                bool   canPassThrough = ( wiLocal.z * woLocal.z > 0.f ) || ( material.specularTransmission > 0.f );

                if ( scatteringPdf > 0.f && canPassThrough && !isnan( wiLocal.x ) && !isnan( wiLocal.y )
                     && !isnan( wiLocal.z ) )
                {
                    const float3 wi = normalize( rotate( wiLocal, conjugate( transformation ) ) );
                    if ( isnan( wi.x ) || isnan( wi.y ) || isnan( wi.z ) )
                        break;
                    optixTrace( handle,
                                pp,
                                wi,
                                0.01f, // tmin
                                1e16f, // tmax
                                0.0f,  // rayTime
                                OptixVisibilityMask( 1 ),
                                OPTIX_RAY_FLAG_NONE,
                                0, // SBT offset
                                1, // SBT stride
                                0, // missSBTIndex
                                shadowPrd.x,
                                shadowPrd.y );
                    if ( !shadowHitInfo.hasHit() )
                    {
                        bsdf *= fabs( wiLocal.z );
                        const float4 environment = rvtx::optix::getEnvironmentColor( parameters.environment, wi, 0.f );

                        const float3 intensity = make_float3( environment );
                        const float  lightPdf  = getPdfEnvironment( parameters.environmentSampling, wi );
                        const float  weight    = powerHeuristic( 1, scatteringPdf, 1, lightPdf );

                        direct += fminf( intensity, bsdf * intensity * weight / fmax( 1e-2f, scatteringPdf ) );
                    }
                }
            }

            finalColor += attenuation * direct;
        }

        // Actual bounce
        float  pdf     = 0.f;
        float3 bsdf    = make_float3( 0.f );
        float3 wiLocal = material.sample( woLocal, hitInfo.t, seed, bsdf, pdf );
        if ( isnan( wiLocal.x ) || isnan( wiLocal.y ) || isnan( wiLocal.z ) || pdf == 0.f )
            break;

        float cosTheta = abs( woLocal.z );
        attenuation *= fminf( make_float3( 1.f ), bsdf * cosTheta / pdf );

        float luminance = getLuminance( attenuation );
        if ( luminance == 0. )
            break;

        // Russian Roulette
        // const float continueProbability = fminf( luminance, 0.95f );
        // if ( rnd( seed ) > continueProbability )
        //     break;
        // attenuation /= continueProbability;

        lastTransmitted = ( woLocal.z * wiLocal.z < 0. );

        ray.direction = rotate( wiLocal, conjugate( transformation ) );
        if ( isnan( ray.direction.x ) || isnan( ray.direction.y ) || isnan( ray.direction.z ) )
            break;
        ray.origin = offsetRay( position, worldNormal * sign( dot( worldNormal, ray.direction ) ) );
        if ( isnan( ray.origin.x ) || isnan( ray.origin.y ) || isnan( ray.origin.z ) )
            break;
    }

    const uint32_t imageIndex       = idx.y * viewSize.x + idx.x;
    float4         accumulatedColor = make_float4( finalColor, alpha );
    if ( subFrameId > 0 )
    {
        const float  weight                   = 1.0f / static_cast<float>( subFrameId + 1 );
        const float4 previousAccumulatedColor = parameters.accumulation[ imageIndex ];
        accumulatedColor                      = lerp( previousAccumulatedColor, accumulatedColor, weight );
    }

    parameters.accumulation[ imageIndex ] = accumulatedColor;

    float3 color                   = make_float3( accumulatedColor );
    accumulatedColor               = make_float4( ACESFilm( color ), accumulatedColor.w );
    parameters.frame[ imageIndex ] = float4ToColor( accumulatedColor );
}

extern "C" __global__ void __miss__general()
{
    const unsigned int u0      = optixGetPayload_0();
    const unsigned int u1      = optixGetPayload_1();
    HitInfo *          hitInfo = u32x2ToType<HitInfo>( make_uint2( u0, u1 ) );
    hitInfo->hit               = false;
}

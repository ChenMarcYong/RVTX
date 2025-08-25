#include "rvtx/optix/material/color.cuh"
#include "rvtx/optix/material/fresnel.cuh"
#include "rvtx/optix/material/lambertian.cuh"
#include "rvtx/optix/material/material.cuh"
#include "rvtx/optix/material/trowbridge_reitz_ggx.cuh"
#include "rvtx/system/material_parameters.hpp"

namespace rvtx::optix
{
    inline Material::Material( const MaterialParameters & mp ) :
        baseColor { mp.baseColor.x, mp.baseColor.y, mp.baseColor.z }, roughness { mp.roughness },
        emissive { mp.emissive.x, mp.emissive.y, mp.emissive.z }, metallic { mp.metallic },
        transmittance { mp.transmittance.x, mp.transmittance.y, mp.transmittance.z }, atDistance( mp.atDistance ),
        ior { mp.ior }, specularTransmission { mp.specularTransmission }, specularTint { mp.specularTint },
        clearcoat { mp.clearcoat }, clearcoatGloss { mp.clearcoat }
    {
    }

    inline __device__ float3 Material::evalDisneyDiffuse( float3 wo, float3 wi ) const
    {
        float alpha = fmax( 1e-4f, roughness * roughness );

        const float3 h   = normalize( wo + wi );
        const float  wih = clamp( dot( wi, h ), 0.f, 1.f );
        const float  won = clamp( abs( wo.z ), 1e-4f, 1.f );
        const float  win = clamp( abs( wi.z ), 1e-4f, 1.f );

        const float fd90 = 0.5f + 2.f * alpha * wih * wih;
        const float f1   = 1.f + ( fd90 - 1.f ) * pow( 1.f - win, 5.f );
        const float f2   = 1.f + ( fd90 - 1.f ) * pow( 1.f - won, 5.f );
        return baseColor * OneOverPi * ( 1.f - metallic ) * f1 * f2;
    }

    inline __device__ float3 Material::sampleDisneyDiffuse( float3 wo, const float2 u ) const
    {
        float3 wi = normalize( cuda::randomInCosineWeightedHemisphere( u ) );
        if ( wo.z < 0.f )
            wi.z *= -1.f;
        return wi;
    }

    inline __device__ float Material::getPdfDisneyDiffuse( float3 wo, float3 wi ) const
    {
        return wo.z * wi.z > 0.f ? fabs( wi.z ) * OneOverPi : 0.f;
    }

    // Linear interpolation between Fresnel metallic and dielectric based on
    // material.metallic.
    // Found: https://schuttejoe.github.io/post/disneybsdf/
    __device__ inline float3 Material::getDisneyFresnel( float3 wo, float3 wi, float3 h ) const
    {
        float  luminance = getLuminance( baseColor );
        float3 tint      = luminance > 0.f ? baseColor * ( 1.f / luminance ) : make_float3( 1.f );

        const float3 baseR0 = make_float3( iorToReflectance( ior ) );
        float3       r0     = lerp( baseR0, tint, specularTint );
        r0                  = lerp( r0, baseColor, metallic );

        const float wih = clamp( fabs( dot( wi, h ) ), 1e-4f, 1.f );
        const float woh = clamp( fabs( dot( wo, h ) ), 1e-4f, 1.f );

        const float3 dielectricF = fresnelSchlick( baseR0, woh );
        const float3 metallicF   = fresnelSchlick( r0, wih );
        return lerp( dielectricF, metallicF, metallic );
    }

    __device__ inline float Material::evalSpecularReflection( float3 wo, float3 wi ) const
    {
        const float r      = fmax( 1e-4f, roughness );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        const float3 h   = normalize( wo + wi );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );

        const float g = getSmithG2GGX( won, win, alpha2 );
        const float d = getDGGX( hn, alpha2 );

        return g * d;
    }

    __device__ inline float Material::getPdfSpecularReflection( float3 wo, float3 wi ) const
    {
        const float r      = fmax( 1e-4f, roughness );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        const float3 h   = normalize( wo + wi );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );
        const float  wih = clamp( dot( wi, h ), 1e-4f, 1.f );

        const float g1 = getSmithG1GGX( wih, alpha2 );
        const float d  = getDGGX( hn, alpha2 );

        // Pdf of the VNDF times the Jacobian of the reflection operator
        return d * g1 * wih / fmax( 1e-4f, 4.f * win * wih );
    }

    __device__ inline float Material::evalSpecularTransmission( float3 wo, float3 wi ) const
    {
        const float r      = fmax( 1e-4f, roughness );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        float inside   = copysignf( 1.f, wo.z );
        bool  isInside = inside < 0.f;

        const float AirIOR = 1.f;
        const float etaI   = isInside ? AirIOR : ior;
        const float etaT   = isInside ? ior : AirIOR;

        const float3 h   = normalize( make_float3( 0.f ) - ( etaI * wi + etaT * wo ) );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  woh = clamp( fabs( dot( wo, h ) ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );
        const float  wih = clamp( fabs( dot( wi, h ) ), 1e-4f, 1.f );

        const float g2 = getSmithG1GGX( wih, alpha2 ) * getSmithG1GGX( woh, alpha2 );
        const float d  = getDGGX( hn, alpha2 );
        const float w  = wih * woh / fmax( 1e-4f, win * won );
        const float s  = etaI * wih + etaT * woh;

        return w * etaT * etaT * g2 * d / fmax( 1e-4f, s * s );
    }

    __device__ inline float Material::getPdfSpecularTransmission( float3 wo, float3 wi ) const
    {
        const float r      = fmax( 1e-4f, roughness );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        float inside   = copysignf( 1.f, wo.z );
        bool  isInside = inside < 0.f;

        const float AirIOR = 1.f;
        const float etaI   = isInside ? AirIOR : ior;
        const float etaT   = isInside ? ior : AirIOR;

        const float3 h   = normalize( make_float3( 0.f ) - ( etaI * wi + etaT * wo ) );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  woh = clamp( fabs( dot( wo, h ) ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );
        const float  wih = clamp( fabs( dot( wi, h ) ), 1e-4f, 1.f );

        const float g1 = getSmithG1GGX( wih, alpha2 );
        const float d  = getDGGX( hn, alpha2 );

        const float s                    = etaI * wih + etaT * woh;
        const float transmissionJacobian = etaT * etaT * woh / fmax( 1e-4f, s * s );
        const float vndf                 = g1 * wih * d / win;

        return transmissionJacobian * vndf;
    }

    __device__ inline float Material::getClearCoatRoughness() const { return 0.6f * ( 1.f - clearcoatGloss ); }

    __device__ inline float Material::evalClearCoat( float3 wo, float3 wi ) const
    {
        const float r      = fmax( 1e-4f, getClearCoatRoughness() );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        const float3 h   = normalize( wo + wi );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );

        const float g = getSmithG2GGX( won, win, alpha2 );
        const float d = getDGGX( hn, alpha2 );

        return clearcoat * 0.25f * g * d;
    }

    __device__ inline float Material::getPDFClearCoat( float3 wo, float3 wi ) const
    {
        float r      = fmax( 1e-4f, getClearCoatRoughness() );
        float alpha  = fmax( 1e-4f, r * r );
        float alpha2 = fmax( 1e-4f, alpha * alpha );

        const float3 h   = normalize( wo + wi );
        const float  hn  = clamp( fabs( h.z ), 1e-4f, 1.f );
        const float  won = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const float  win = clamp( fabs( wi.z ), 1e-4f, 1.f );
        const float  wih = clamp( dot( wi, h ), 1e-4f, 1.f );

        const float g1 = getSmithG1GGX( wih, alpha2 );
        const float d  = getDGGX( hn, alpha2 );

        // Pdf of the VNDF times the Jacobian of the reflection operator
        return clearcoat * 0.25f * d * g1 * wih / fmax( 1e-4f, 4.f * win * wih );
    }

    __device__ inline float3 Material::evaluate( const float3 wo, const float3 wi, const float t ) const
    {
        const float win       = clamp( fabs( wi.z ), 1e-4f, 1.f );
        const float won       = clamp( fabs( wo.z ), 1e-4f, 1.f );
        const bool  entering  = wi.z > 0.f;
        const bool  doReflect = wi.z * wo.z > 0.f;

        float3 weight = make_float3( 1.f );
        if ( !entering && specularTransmission > 0.f && atDistance > 0.f )
        {
            const float3 temp = make_float3( //
                logf( transmittance.x ),
                logf( transmittance.y ),
                logf( transmittance.z ) );
            weight *= make_float3( expf( temp.x ), expf( temp.y ), expf( temp.z ) ) * abs( t ) / atDistance;
        }

        const float AirIOR = 1.f;
        float       etaI   = entering ? ior : AirIOR;
        float       etaT   = entering ? AirIOR : ior;

        if ( doReflect )
        {
            const float3 h = normalize( wi + wo );
            const float3 f = getDisneyFresnel( wi, wo, h );

            const float r      = fmax( 1e-4f, roughness );
            const float alpha  = fmax( 1e-4f, r );
            const float alpha2 = fmax( 1e-4f, alpha * alpha );

            const float nh = clamp( abs( h.z ), 1e-4f, 1.f );
            const float lh = clamp( abs( dot( wi, h ) ), 1e-4f, 1.f );

            const float  diffuseWeight = 1.f - specularTransmission;
            const float3 diffuse       = diffuseWeight * evalDisneyDiffuse( wo, wi );
            const float  specular      = evalSpecularReflection( wo, wi );

            float woh = clamp( abs( dot( wo, h ) ), 1e-4f, 1.f );
            float ccf = fresnelSchlick( make_float3( iorToReflectance( 1.5f ) ), woh ).x;

            return weight * ( ( 1. - f ) * diffuse + f * specular + ccf * evalClearCoat( wo, wi ) );
        }

        const float3 h = normalize( make_float3( 0.f ) - ( etaI * wi + etaT * wo ) );
        const float3 f = getDisneyFresnel( wi, wo, h );

        const float transmissionWeight   = specularTransmission;
        const float specularTransmission = transmissionWeight * evalSpecularTransmission( wo, wi );
        return weight
               * ( make_float3( sqrtf( baseColor.x ), sqrtf( baseColor.y ), sqrtf( baseColor.z ) ) * ( 1.f - f )
                   * specularTransmission );
    }

    __device__ inline float3 Material::sample( float3                    wo,
                                               float                     t,
                                               rvtx::cuda::RandomState & seed,
                                               float3 &                  weight,
                                               float &                   pdf ) const
    {
        const float r      = fmax( 1e-4f, roughness );
        const float alpha  = fmax( 1e-4f, r * r );
        const float alpha2 = fmax( 1e-4f, alpha * alpha );

        float inside   = copysignf( 1.f, wo.z );
        bool  isInside = inside < 0.;

        pdf    = 1.f;
        weight = make_float3( 1.f );
        if ( isInside && specularTransmission > 0.f && atDistance > 0.f )
        {
            const float3 temp = make_float3( //
                logf( transmittance.x ),
                logf( transmittance.y ),
                logf( transmittance.z ) );
            weight *= make_float3( expf( temp.x ), expf( temp.y ), expf( temp.z ) ) * abs( t ) / atDistance;
        }

        float3 h = make_float3( 0., 0., 1. );

        const float4 alea = make_float4(
            rvtx::cuda::rnd( seed ), rvtx::cuda::rnd( seed ), rvtx::cuda::rnd( seed ), rvtx::cuda::rnd( seed ) );
        if ( clearcoat > 0.f )
        {
            const float ccRoughness = getClearCoatRoughness();
            const float ccAlpha     = fmax( 1e-4f, ccRoughness * ccRoughness );
            const float ccAlpha2    = fmax( 1e-4f, ccAlpha * ccAlpha );

            float3 ccH = h;
            if ( ccRoughness > .0f )
                ccH = sampleGGXVNDF( wo, ccAlpha, ccAlpha, alea.z, alea.w );

            const float woh = clamp( fabs( dot( wo, ccH ) ), 1e-4f, 1.f );
            const float ccf = fresnelSchlick( make_float3( iorToReflectance( 1.5f ) ), woh ).x;
            if ( alea.y < clearcoat * ccf )
            {
                const float3 wi = reflect( -wo, ccH );

                const float hn  = clamp( fabs( ccH.z ), 1e-4f, 1.f );
                const float woh = clamp( fabs( dot( wo, ccH ) ), 1e-4f, 1.f );
                const float wih = clamp( fabs( dot( wi, ccH ) ), 1e-4f, 1.f );

                const float g1 = getSmithG1GGX( woh, ccAlpha2 );
                const float g2 = getSmithG1GGX( wih, ccAlpha2 ) * g1;
                weight *= g2 / fmax( 1e-4f, g1 );
                pdf *= 1.f;
                return wi;
            }
        }

        if ( roughness > 0.f )
            h = sampleGGXVNDF( wo, alpha, alpha, alea.z, alea.w );

        const float3 f              = getDisneyFresnel( wo, wo, h );
        const float  specularWeight = length( f );
        const bool   fullSpecular   = roughness == 0.f && metallic == 1.f;
        const float  type           = fullSpecular ? 0.f : alea.x;

        if ( type < specularWeight )
        {
            float3 wi = reflect( -wo, h );

            weight *= f * evalSpecularReflection( wo, wi );
            pdf *= getPdfSpecularReflection( wo, wi );
            pdf *= fullSpecular ? 1.f : specularWeight;
            return wi;
        }

        const float transmissionType           = type - specularWeight;
        const float specularTransmissionWeight = ( 1.f - specularWeight ) * specularTransmission;
        if ( transmissionType < specularTransmissionWeight )
        {
            const float  AirIOR = 1.f;
            const float  etaI   = isInside ? ior : AirIOR;
            const float  etaT   = isInside ? AirIOR : ior;
            const float3 wi     = rvtx::cuda::refract( -wo, h, etaI / etaT );

            weight *= make_float3( sqrtf( baseColor.x ), sqrtf( baseColor.y ), sqrtf( baseColor.z ) ) * ( 1.f - f )
                      * evalSpecularTransmission( wo, wi );
            pdf *= getPdfSpecularTransmission( wo, wi );

            return wi;
        }

        const float3 wi = sampleDisneyDiffuse( wo, make_float2( cuda::rnd( seed ), cuda::rnd( seed ) ) );
        weight          = ( 1. - f ) * evalDisneyDiffuse( wo, wi );
        pdf             = getPdfDisneyDiffuse( wo, wi );

        return wi;
    }

    __device__ inline float Material::getPdfMaterial( float3 wo, float3 wi, rvtx::cuda::RandomState & seed ) const
    {
        const float3 r0 = lerp( make_float3( iorToReflectance( ior ) ), baseColor, metallic );

        const float3 h = normalize( wi + wo );
        const float3 f = fresnelSchlick( r0, fabs( dot( wo, h ) ) );

        const float specularWeight = length( f );
        const bool  fullSpecular   = roughness == 0.f && metallic == 1.f;
        const float type           = fullSpecular ? 0.f : cuda::rnd( seed );
        if ( type < specularWeight )
            return getPdfSpecularReflection( wo, wi ) * ( fullSpecular ? 1.f : specularWeight );

        float transmissionType           = type - specularWeight;
        float specularTransmissionWeight = ( 1.f - specularWeight ) * specularTransmission;
        if ( transmissionType < specularTransmissionWeight )
            return specularTransmission * ( 1.f - specularWeight ) * getPdfSpecularTransmission( wo, wi );

        return getPdfDisneyDiffuse( wo, wi ) * ( 1.f - specularTransmission ) * ( 1.f - specularWeight );
    }
} // namespace rvtx::optix

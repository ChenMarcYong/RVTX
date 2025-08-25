#ifndef RVTX_DIL_TYPES_HPP
#define RVTX_DIL_TYPES_HPP


namespace rvtx::dil {

    struct Sphere
    {
        Diligent::float3 position;
        float radius;
        Diligent::float3 color;
        float visibility;
    };

    struct SphereSettings
    {
        Diligent::float4x4 uMVMatrix;
        Diligent::float4x4 uProjMatrix;
        float uRadiusAdd;
        Diligent::Uint32 uIsPerspective;
        Diligent::float2 _padding = Diligent::float2(0, 0);
        Diligent::float2   _pad2;
    };


    //LinearizeDepthPostProcess

    struct alignas(16) CameraCBData
    {
        float uClipInfo[4]; // near, far, fpn, nf
        uint32_t uIsPerspective;
        uint32_t _pad[3];   // padding pour rester 16-bytes aligned (std140 côté GL)
    };

}

#endif
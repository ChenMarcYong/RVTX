cbuffer SphereSettings : register(b0)
{
    float4x4 uMVMatrix;
    float4x4 uProjMatrix;
    float uRadiusAdd;
    uint uIsPerspective;
    float2 _padding;
};

struct VS_OUT
{
    float4 SVPos : SV_POSITION;
    float3 vViewSpherePos : TEXCOORD0;
    float3 vSphereColor : TEXCOORD1;
    float vSphereRad : TEXCOORD2;
    float vVisible : TEXCOORD3;
    float3 vImpU : TEXCOORD4;
    float3 vImpV : TEXCOORD5;
    float vDotViewSpherePos : TEXCOORD6;
    nointerpolation uint vId : TEXCOORD7;
};

struct GS_OUT
{
    float4 Position : SV_POSITION;
    float3 viewImpPos : TEXCOORD0;
    float3 viewSpherePos : TEXCOORD1;
    float3 sphereColor : TEXCOORD2;
    float sphereRad : TEXCOORD3;
    float dotViewSpherePos : TEXCOORD4;
    nointerpolation uint id : TEXCOORD7;
};

[maxvertexcount(4)]
void main(point VS_OUT vsIn[1], inout TriangleStream<GS_OUT> triStream)
{
    if (vsIn[0].vVisible == 0)
        return;

    float3 center = vsIn[0].vViewSpherePos;
    float3 U = vsIn[0].vImpU;
    float3 V = vsIn[0].vImpV;

    // Coins du quad
    float3 quadPos[4] =
    {
        center - U - V,
        center + U - V,
        center - U + V,
        center + U + V
    };

    for (int i = 0; i < 4; ++i)
    {
        GS_OUT outVert;
        outVert.Position = mul(float4(quadPos[i], 1.0f), uProjMatrix);

        outVert.viewImpPos = quadPos[i];
        outVert.viewSpherePos = center;
        outVert.sphereColor = vsIn[0].vSphereColor;
        outVert.sphereRad = vsIn[0].vSphereRad;
        outVert.dotViewSpherePos = vsIn[0].vDotViewSpherePos;
        outVert.id = vsIn[0].vId;

        triStream.Append(outVert);
    }

    triStream.RestartStrip();
}

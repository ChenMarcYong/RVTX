cbuffer SphereSettings : register(b0)
{
    float4x4 uMVMatrix; // View (ou ModelView) Matrix
    float4x4 uProjMatrix; // Projection Matrix
    float uRadiusAdd;
    uint uIsPerspective;
    float2 _padding;
};

struct Sphere
{
    float3 position;
    float radius;
    float3 color;
    float visibility;
};

StructuredBuffer<Sphere> spheres : register(t0);
StructuredBuffer<uint> ids : register(t1);

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

VS_OUT main(uint VertexID : SV_VertexID)
{
    VS_OUT vsOut;

    Sphere sphere = spheres[VertexID];

    // Transformation en coordonnées vue
    float4 viewPos4 = mul(float4(sphere.position, 1.0f), uMVMatrix);
    float3 viewPos = viewPos4.xyz;

    vsOut.vViewSpherePos = viewPos;
    vsOut.vSphereRad = sphere.radius + uRadiusAdd;
    vsOut.vSphereColor = sphere.color;
    vsOut.vVisible = (sphere.visibility > 0.5f) ? 1.0f : 0.0f;
    vsOut.vId = ids[VertexID];

    float dist2 = dot(viewPos, viewPos);
    float dist = sqrt(dist2);
    dist = max(dist, 1e-4f);

    float3 viewDir = viewPos / dist;
    vsOut.vDotViewSpherePos = dist2;

    // Vecteurs pour le quad
    float3 up = abs(viewDir.y) < 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 u = normalize(cross(viewDir, up));
    float3 v = cross(u, viewDir);

    float impHalfSize = vsOut.vSphereRad;
    if (uIsPerspective != 0)
    {
        float sinAngle = vsOut.vSphereRad / dist;
        float angle = asin(saturate(sinAngle));
        impHalfSize = tan(angle) * dist;
    }

    vsOut.vImpU = u * impHalfSize;
    vsOut.vImpV = v * impHalfSize;

    vsOut.SVPos = mul(viewPos4, uProjMatrix);
    return vsOut;
}

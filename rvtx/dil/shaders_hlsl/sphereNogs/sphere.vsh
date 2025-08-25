// sphere_impostor_instanced.vsh
cbuffer SphereSettings : register(b0)
{
    float4x4 uMVMatrix; // view (déjà ce que tu charges en C++)
    float4x4 uProjMatrix; // projection
    float uRadiusAdd;
    uint uIsPerspective; // 0=ortho, 1=persp
    float2 _pad;
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
    float4 pos : SV_Position; // clip-space
    float3 viewImpPos : TEXCOORD0; // position du sommet du quad en espace vue
    float3 viewCenter : TEXCOORD1; // centre sphère en espace vue
    float3 color : TEXCOORD2;
    float radius : TEXCOORD3;
    float visible : TEXCOORD4;
    uint id : TEXCOORD5;
};

static const float2 kOffsets[4] =
{
    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
};

VS_OUT main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VS_OUT o;

    Sphere s = spheres[instanceID];

    // IMPORTANT : ordre de mul() aligné avec ton code C++ (tu utilisais déjà mul(uMVMatrix, v))
    float4 viewCenter4 = mul(float4(s.position, 1.0), uMVMatrix);
    float3 centerV = viewCenter4.xyz;

    float rad = s.radius + uRadiusAdd;

    // Base billboard dans l'espace vue
    float dist = max(length(centerV), 1e-4);
    float3 viewDir = centerV / dist;
    float3 up = (abs(viewDir.y) < 0.99) ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 U = normalize(cross(viewDir, up));
    float3 V = cross(U, viewDir);

    // Taille du quad (persp vs ortho) en unités VUE
    float halfSize = rad;
    if (uIsPerspective != 0u)
    {
        float sinA = saturate(rad / dist);
        float ang = asin(sinA);
        halfSize = tan(ang) * dist;
    }

    float2 offs = kOffsets[vertexID & 3u];
    float3 quadV = centerV + U * (offs.x * halfSize) + V * (offs.y * halfSize);

    o.viewImpPos = quadV;
    o.viewCenter = centerV;
    o.color = s.color;
    o.radius = rad;
    o.visible = (s.visibility > 0.5f) ? 1.0f : 0.0f;
    o.id = ids[instanceID];

    // clip
    o.pos = mul(float4(quadV, 1.0), uProjMatrix);
    return o;
}

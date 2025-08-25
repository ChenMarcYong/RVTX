// ======================================================
// HLSL (SM5+)
// Geometry Shader: lines -> triangle strip (4 vertices)
// ======================================================

struct GSIn
{
    // NOTE: le VS sort la position en espace VUE dans SV_Position,
    // comme dans ton GLSL (gl_Position = viewPos).
    float4 pos : SV_Position; // view-space position
    float3 vVertexColor : TEXCOORD0; // flat
    uint vVertexVis : TEXCOORD1; // flat
    uint vId : TEXCOORD2; // flat
};

struct PSIn
{
    float4 pos : SV_Position; // clip-space position (après uProjMatrix)
    float3 vVertexColor : TEXCOORD0;
    uint vVertexVis : TEXCOORD1;
    uint vId : TEXCOORD2;
};

// --- UBO équivalent (std140 -> cbuffer) ---
cbuffer CylinderSettings : register(b0)
{
    float4x4 uMVMatrix;
    float4x4 uProjMatrix;
    float uCylRadius;
    uint uIsPerspective; // bool -> uint pour l’alignement
    float2 _padCB;
}

static PSIn MakeVtx(float3 viewPos, const GSIn base)
{
    PSIn o;
    o.pos = mul(uProjMatrix, float4(viewPos, 1.0));
    o.vVertexColor = base.vVertexColor;
    o.vVertexVis = base.vVertexVis;
    o.vId = base.vId;
    return o;
}

[maxvertexcount(4)]
void main(line GSIn inputVerts[2], inout TriangleStream<PSIn> triStream)
{
    // Récupère les deux extrémités du segment en espace vue
    float3 p0 = inputVerts[0].pos.xyz;
    float3 p1 = inputVerts[1].pos.xyz;

    // On propage les attributs "flat" depuis le 1er sommet (comme en GLSL)
    GSIn base = inputVerts[0];

    if (uIsPerspective != 0u)
    {
        // Billboard en perspective : base sur le vecteur vue
        float3 view = normalize((p0 + p1) * 0.5f);
        float3 z = normalize(p1 - p0);
        float3 x = normalize(cross(view, z));
        float3 y = cross(x, z); // pas besoin de normaliser

        float3 v1 = p0 + x * uCylRadius;
        float3 v2 = p0 - x * uCylRadius;
        float3 v3 = p1 + x * uCylRadius;
        float3 v4 = p1 - x * uCylRadius;

        triStream.Append(MakeVtx(v1, base));
        triStream.Append(MakeVtx(v2, base));
        triStream.Append(MakeVtx(v3, base));
        triStream.Append(MakeVtx(v4, base));
        triStream.RestartStrip();
    }
    else
    {
        // Ortho : décalage perpendiculaire dans le plan XY (z inchangé)
        float3 dirCyl = normalize(p1 - p0);
        float3 vertStep = normalize(float3(-dirCyl.y, dirCyl.x, 0.0)) * uCylRadius;

        float3 v1 = p0 + vertStep;
        float3 v2 = p0 - vertStep;
        float3 v3 = p1 + vertStep;
        float3 v4 = p1 - vertStep;

        triStream.Append(MakeVtx(v1, base));
        triStream.Append(MakeVtx(v2, base));
        triStream.Append(MakeVtx(v3, base));
        triStream.Append(MakeVtx(v4, base));
        triStream.RestartStrip();
    }
}

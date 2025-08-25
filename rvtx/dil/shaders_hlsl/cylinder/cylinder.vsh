// ======================================================
// HLSL (SM5+)
// Traduction d’un VS GLSL utilisant gl_VertexID,
// UBO std140 et StructuredBuffers
// ======================================================

struct Sphere
{
	float3 position; // 12
	float radius; // 16
	float3 color; // 28
	float visibility; // 32  (taille totale: 32 octets)
};

// Remplace le UBO std140 "CylinderSettings" (binding = 0)
cbuffer CylinderSettings : register(b0)
{
	float4x4 uMVMatrix;
	float4x4 uProjMatrix;
	float uCylRadius;
	uint uIsPerspective; 
	float2 _padding; // padding pour alignement 16 octets
};

// Buffers équivalents aux SSBO/UBO GLSL
// NOTE: adaptez les slots t# selon votre pipeline.
StructuredBuffer<Sphere> spheres : register(t0);
StructuredBuffer<uint> sphereIndices : register(t1);
StructuredBuffer<uint> ids : register(t2);

// Sorties du VS
struct VSOut
{
	float4 pos : SV_Position;
	float3 vVertexColor : TEXCOORD0;
	uint vVertexVis : TEXCOORD1;
	uint vId : TEXCOORD2;
};

VSOut main(uint vertexID : SV_VertexID)
{
	VSOut o;

    // Equiv: Sphere sphere = spheres[sphereIndices[gl_VertexID]];
	Sphere sphere = spheres[sphereIndices[vertexID]];
	
	o.vVertexColor = sphere.color;
	
	o.vVertexVis = (uint) sphere.visibility;
	
	o.vId = ids[vertexID >> 1];
	
	float4 viewPos = mul(uMVMatrix, float4(sphere.position, 1.0));
	o.pos = viewPos;

	return o;
}

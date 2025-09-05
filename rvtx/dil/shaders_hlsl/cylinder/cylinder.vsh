// ======================================================
// HLSL (SM5+)
// Traduction d’un VS GLSL utilisant gl_VertexID,
// UBO std140 et StructuredBuffers
// ======================================================

struct Sphere
{
	float3 position;
	float radius;
	float3 color;
	float visibility;
};

cbuffer CylinderSettings : register(b0)
{
	float4x4 uMVMatrix;
	float4x4 uProjMatrix;
	float uCylRadius;
	uint uIsPerspective; 
	float2 _padding;
};
StructuredBuffer<Sphere> spheres : register(t0);
StructuredBuffer<uint> sphereIndices : register(t1);
StructuredBuffer<uint> ids : register(t2);

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
	
	Sphere sphere = spheres[sphereIndices[vertexID]];
	
	o.vVertexColor = sphere.color;
	
	o.vVertexVis = (uint) sphere.visibility;
	
	o.vId = ids[vertexID >> 1];
	
	float4 viewPos = mul(uMVMatrix, float4(sphere.position, 1.0));
	o.pos = viewPos;

	return o;
}

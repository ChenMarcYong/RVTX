#version 450

struct Sphere
{
    vec3  position;
    float radius;
    vec3  color;
    float visibility;
};

layout(std140, binding = 0) uniform CylinderSettings
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	float uCylRadius;
	bool  uIsPerspective;
};

layout(std140, binding = 1) buffer Spheres
{
    Sphere spheres[];
};

layout(std430, binding = 3) buffer Cylinder
{
    uint sphereIndices[];
};

layout(std430, binding = 4) buffer SpheresIDs
{
    uint ids[];
};


flat out vec3 vVertexColor;
flat out uint vVertexVis;
flat out uint vId;

void main()
{
	Sphere sphere  = spheres[sphereIndices[gl_VertexID]];

	vVertexColor = sphere.color;
	vVertexVis	 = uint(sphere.visibility);

	vId = ids[gl_VertexID >> 1];

	// Vertex position in view space.
	gl_Position = uMVMatrix * vec4( sphere.position, 1. );
}

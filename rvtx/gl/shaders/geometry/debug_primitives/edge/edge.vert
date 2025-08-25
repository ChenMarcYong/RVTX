#version 450

struct EdgeParams
{
    vec3  color;
    float radius;
};

layout(std140, binding = 0) uniform Camera
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
};

layout(std430, binding = 1) buffer NodesPositions
{
    vec4 nodesPositions[];
};

layout(std430, binding = 3) buffer Edge
{
    uint nodesIndices[];
};

layout(std430, binding = 4) buffer EdgesParams
{
    EdgeParams edgeParams[];
};


flat out vec3  vEdgeColor;
flat out float vEdgeRadii;

void main()
{
	EdgeParams edge  = edgeParams[gl_VertexID >> 1];

	vEdgeColor = edge.color;
	vEdgeRadii = edge.radius;

	// Vertex position in view space.
	gl_Position = uMVMatrix * vec4( nodesPositions[nodesIndices[gl_VertexID]].xyz, 1. );
}

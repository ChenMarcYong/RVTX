#version 450

 struct Vertex
{
    vec4 position;
    vec4 normal;
    vec4 color;
};

layout(std140, binding = 0) uniform VertexSettings
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
	bool  uUseSingleId;
};

layout(std140, binding = 1) buffer Vertices
{
    Vertex vertices[];
};

out vec3 vViewPos;
out vec3 vNormal;
out vec3 vColor;

void main()
{
	vec4 vertexViewPosition = uMVMatrix * vertices[gl_VertexID].position;

	gl_Position  = uProjMatrix * vertexViewPosition;

	vViewPos = vertexViewPosition.xyz;
	vNormal = normalize(vec3(transpose(inverse(uMVMatrix)) * vertices[gl_VertexID].normal));
	vColor = vec3(vertices[gl_VertexID].color);
}

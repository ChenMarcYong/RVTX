#version 450

struct NodeData
{
    vec3  position;
    float radius;
};

struct NodeParams
{
    vec3  color;
    float visible;
};

layout(std140, binding = 0) uniform Camera
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
};

layout(std430, binding = 1) buffer NodesPositions
{
    NodeData nodesPositions[];
};

layout(std430, binding = 2) buffer NodesParams
{
    NodeParams nodesParams[];
};

flat out vec3  vViewNodePos; // Node position in view space.
flat out vec3  vNodeColor;
flat out float vNodeRad;
flat out uint  vVisible;
flat out vec3  vImpU; // Impostor vectors.
flat out vec3  vImpV;
flat out float vDotViewNodePos;

void main()
{
	const NodeParams nodeParams  = nodesParams[gl_VertexID];
	const NodeData nodeData = nodesPositions[gl_VertexID];
	vViewNodePos   = vec3( uMVMatrix * vec4( nodeData.position, 1. ) );
	vNodeColor     = nodeParams.color;
	vNodeRad	   = nodeData.radius;
	vVisible	   = uint(nodeParams.visible);

	// Compute normalized view vector.
	vDotViewNodePos		  = dot( vViewNodePos, vViewNodePos );
	const float dNodeCenter = sqrt( vDotViewNodePos );
	const vec3	view		  = vViewNodePos / dNodeCenter;

	if (uIsPerspective)
	{
		// Impostor in front of the sphere.
		vec3 viewImpPos = vViewNodePos - vNodeRad * view;

		// Compute impostor size.
		const float sinAngle = vNodeRad / dNodeCenter;
		const float tanAngle = tan( asin( sinAngle ) );
		const float impSize	 = tanAngle * length( viewImpPos );

		// Compute impostor vectors.
		// TODO: simplify normalize ? (vImpU.x == 0) but normalize should be hard optimized on GPU...
		// But for cross always better doing no calculation.
		// vImpU = normalize( cross( dir, vec3( 1.f, 0.f, 0.f ) ) ); becomes:
		vImpU = normalize( vec3( 0.f, view.z, -view.y ) );
		// TODO: simplify cross ? (vImpU.x == 0) but cross should be hard optimized on GPU...
		vImpV = cross( vImpU, view ) * impSize; // No need to normalize.
		vImpU *= impSize;

		gl_Position = vec4( viewImpPos, 1.f );
	}
	else // Orthographic
	{
		vImpU = vec3(-1, 0, 0) * vNodeRad;
		vImpV = vec3(0, -1, 0) * vNodeRad; 

		gl_Position = vec4( vViewNodePos + vec3(0, 0, vNodeRad), 1.f );
	}
}

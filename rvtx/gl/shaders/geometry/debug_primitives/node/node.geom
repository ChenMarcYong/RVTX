#version 450

layout( points ) in;
layout( triangle_strip, max_vertices = 4 ) out;

layout(std140, binding = 0) uniform Camera
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
};

flat in vec3  vViewNodePos[]; // Node position in view space.
flat in vec3  vNodeColor[];
flat in float vNodeRad[];
flat in uint  vVisible[];
// Impostor vectors.
flat in vec3  vImpU[];
flat in vec3  vImpV[];
flat in float vDotViewNodePos[];

smooth out vec3 viewImpPos;	   // Impostor position in view space.
flat out vec3	viewNodePos; // Node position in view space.
flat out vec3	nodeColor;
flat out float	nodeRad;
flat out float	dotViewNodePos;

void emitQuad( const vec3 v1, const vec3 v2, const vec3 v3, const vec3 v4 )
{
	viewImpPos	= v1;
	gl_Position = uProjMatrix * vec4( viewImpPos, 1.f );
	EmitVertex();

	viewImpPos	= v2;
	gl_Position = uProjMatrix * vec4( viewImpPos, 1.f );
	EmitVertex();

	viewImpPos	= v3;
	gl_Position = uProjMatrix * vec4( viewImpPos, 1.f );
	EmitVertex();

	viewImpPos	= v4;
	gl_Position = uProjMatrix * vec4( viewImpPos, 1.f );
	EmitVertex();

	EndPrimitive();
}

void main()
{
	// Output data.
	viewNodePos	   = vViewNodePos[ 0 ];
	nodeColor      = vNodeColor[ 0 ];
	nodeRad		   = vNodeRad[ 0 ];
	dotViewNodePos = vDotViewNodePos[ 0 ];

	// Compute impostors vertices.
	const vec3 v1 = gl_in[ 0 ].gl_Position.xyz - vImpU[ 0 ] - vImpV[ 0 ];
	const vec3 v2 = gl_in[ 0 ].gl_Position.xyz + vImpU[ 0 ] - vImpV[ 0 ];
	const vec3 v3 = gl_in[ 0 ].gl_Position.xyz - vImpU[ 0 ] + vImpV[ 0 ];
	const vec3 v4 = gl_in[ 0 ].gl_Position.xyz + vImpU[ 0 ] + vImpV[ 0 ];

	emitQuad( v1, v2, v3, v4 );
}

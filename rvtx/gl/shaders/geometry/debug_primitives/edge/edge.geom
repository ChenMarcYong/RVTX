#version 450

layout( lines ) in;
layout( triangle_strip, max_vertices = 4 ) out;

layout(std140, binding = 0) uniform Camera
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
};

flat in vec3  vEdgeColor[];
flat in float vEdgeRadii[];

smooth out vec3 viewImpPos;		  // Impostor position in view space.
flat out vec3	viewEdgeVert[ 2 ]; // Edge vertices position in view space.
flat out vec3	colors[ 2 ];
flat out float	radii[ 2 ];

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
	viewEdgeVert[ 0 ] = gl_in[ 0 ].gl_Position.xyz;
	viewEdgeVert[ 1 ] = gl_in[ 1 ].gl_Position.xyz;
	colors[ 0 ]		 = vEdgeColor[ 0 ];
	colors[ 1 ]		 = vEdgeColor[ 1 ];
	radii[ 0 ]		 = vEdgeRadii[ 0 ];

	// Flip is vertex 0 is farther than vertex 1.
	vec3 viewImpPos0, viewImpPos1;
	if ( viewEdgeVert[ 0 ].z < viewEdgeVert[ 1 ].z )
	{
		viewImpPos0 = viewEdgeVert[ 1 ];
		viewImpPos1 = viewEdgeVert[ 0 ];
	}
	else
	{
		viewImpPos0 = viewEdgeVert[ 0 ];
		viewImpPos1 = viewEdgeVert[ 1 ];
	}

	if (uIsPerspective)
	{
		// Compute normalized view vector to cylinder center.
		const vec3 view = normalize( ( viewImpPos0 + viewImpPos1 ) * 0.5f );

		// Compute cylinder coordinates system with 'x' orthogonal to 'view'.
		const vec3 z = normalize( viewImpPos1 - viewImpPos0 );
		const vec3 x = normalize( cross( view, z ) );
		const vec3 y = cross( x, z ); // no need to normalize

		// Compute impostor construction vectors.
		const float dV0 = length( viewImpPos0 );
		const float dV1 = length( viewImpPos1 );

		const float sinAngle = radii[ 0 ] / dV0;
		float		angle	 = asin( sinAngle );
		const vec3	y1		 = y * radii[ 0 ];
		const vec3	x2		 = x * radii[ 0 ] * cos( angle );
		const vec3	y2		 = y1 * sinAngle;
		angle				 = asin( radii[ 0 ] / dV1 );
		const vec3 x3		 = x * ( dV1 - radii[ 0 ] ) * tan( angle );

		// Compute impostors vertices.
		const vec3 v1 = viewImpPos0 - x2 + y2;
		const vec3 v2 = viewImpPos0 + x2 + y2;
		const vec3 v3 = viewImpPos1 - x3 + y1;
		const vec3 v4 = viewImpPos1 + x3 + y1;

		emitQuad( v1, v2, v3, v4 );
	}
	else
	{
		const vec3 dirCyl   = normalize( viewImpPos1 - viewImpPos0 );
		const vec3 vertStep = normalize(vec3(-dirCyl.y, dirCyl.x, 0)) * radii[ 0 ];

		// Compute impostors vertices.
		const vec3 v1 = viewImpPos0 + vertStep;
		const vec3 v2 = viewImpPos0 - vertStep;
		const vec3 v3 = viewImpPos1 + vertStep;
		const vec3 v4 = viewImpPos1 - vertStep;

		emitQuad( v1, v2, v3, v4 );
	}
}

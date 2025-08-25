#version 450

//#define SHOW_IMPOSTORS

layout(std140, binding = 0) uniform Camera
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
};

smooth in vec3 viewImpPos;
flat in vec3   viewEdgeVert[ 2 ];
flat in vec3   colors[ 2 ];
flat in float  radii[ 2 ];

// 3 16 bits for position.
// 3 16 bits for normal.
// 1 32 bits for padding.
layout( location = 0 ) out uvec4 outViewPositionNormal;
// 3 32 bits for color.
// 1 32 bits for specular.
layout( location = 1 ) out vec4 outColor;

float computeDepth( const vec3 v )
{
	// Computes 'v' NDC depth ([-1,1]).
	const float ndcDepth = ( v.z * uProjMatrix[ 2 ].z + uProjMatrix[ 3 ].z ) / (uIsPerspective ? -v.z : 1.0);
	// Return depth according to depth range.
	return ( gl_DepthRange.diff * ndcDepth + gl_DepthRange.near + gl_DepthRange.far ) * 0.5f;
}

void main()
{
	// Only consider cylinder body.
	const vec3 v1v0	  = viewEdgeVert[ 1 ] - viewEdgeVert[ 0 ];
	const vec3 v0	  = (uIsPerspective ? vec3(0) : viewImpPos) - viewEdgeVert[ 0 ];
	const vec3 rayDir = (uIsPerspective ? normalize( viewImpPos ) : vec3(0, 0, -1));

	const float d0 = dot( v1v0, v1v0 );
	const float d1 = dot( v1v0, rayDir );
	const float d2 = dot( v1v0, v0 );

	const float a = d0 - d1 * d1;
	const float b = d0 * dot( v0, rayDir ) - d2 * d1;
	const float c = d0 * dot( v0, v0 ) - d2 * d2 - radii[0] * radii[0] * d0;

	const float h = b * b - a * c;

	if ( h < 0.f )
	{
#ifdef SHOW_IMPOSTORS
		// Show impostors for debugging purpose.
		uvec4 colorNormal = uvec4( 0 );
		// Fill G-buffers.
		uvec4 viewPositionNormalCompressed;
		viewPositionNormalCompressed.x = packHalf2x16( viewImpPos.xy );
		viewPositionNormalCompressed.y = packHalf2x16( vec2( viewImpPos.z, -rayDir.x ) );
		viewPositionNormalCompressed.z = packHalf2x16( -rayDir.yz );
		viewPositionNormalCompressed.w = 0; // Padding.

		// Output data.
		outViewPositionNormal = viewPositionNormalCompressed;
		outColor			  = vec4( 1.f, 0.f, 0.f, 32.f ); // w = specular shininess.

		gl_FragDepth = computeDepth( viewImpPos );
#else
		discard;
#endif
	}
	else
	{
		// Solve equation (only first intersection).
		const float t = ( -b - sqrt( h ) ) / a;

		const float y = d2 + t * d1;

		if ( y < 0.f || y > d0 )
		{
#ifdef SHOW_IMPOSTORS
			// fill G-buffers
			uvec4 viewPositionNormalCompressed;
			viewPositionNormalCompressed.x = packHalf2x16( viewImpPos.xy );
			viewPositionNormalCompressed.y = packHalf2x16( vec2( viewImpPos.z, -rayDir.x ) );
			viewPositionNormalCompressed.z = packHalf2x16( -rayDir.yz );
			viewPositionNormalCompressed.w = 0; // Padding.

			// Output data.
			outViewPositionNormal = viewPositionNormalCompressed;
			outColor			  = vec4( 1.f, 0.f, 0.f, 32.f ); // w = specular shininess.

			gl_FragDepth = computeDepth( viewImpPos );
#else
			discard;
#endif
		}
		else
		{
			// Compute hit point and normal (always in view space).
			vec3 hit;
			vec3 normal;

			if (uIsPerspective)
			{
				hit	 = rayDir * t;
				normal = normalize( v0 + hit - v1v0 * y / d0 );
			}
			else
			{
				hit	  = viewImpPos + vec3(0, 0, -t);
				normal = normalize( hit - viewEdgeVert[ 0 ] - v1v0 * y / d0 );
			}

			// Fill depth buffer.
			gl_FragDepth = computeDepth( hit );

			// Color with good color extremity.
			const vec3 color = colors[ int( y > d0 * 0.5f ) ];

			// Compress color and normal.
			uvec4 viewPositionNormalCompressed;
			viewPositionNormalCompressed.x = packHalf2x16( hit.xy );
			viewPositionNormalCompressed.y = packHalf2x16( vec2( hit.z, normal.x ) );
			viewPositionNormalCompressed.z = packHalf2x16( normal.yz );
			viewPositionNormalCompressed.w = 0; // Padding.

			// Output data.
			outViewPositionNormal = viewPositionNormalCompressed;
			outColor			  = vec4( color, 32.f ); // w = specular shininess.
		}
	}
}

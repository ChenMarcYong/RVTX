#version 450

//#define SHOW_IMPOSTORS

layout (depth_greater) out float gl_FragDepth;

layout(std140, binding = 0) uniform SphereSettings
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	float uRadiusAdd;
	bool  uIsPerspective;
};

smooth in vec3 viewImpPos;
flat in vec3   viewSpherePos;
flat in vec3   sphereColor;
flat in float  sphereRad;
flat in float  dotViewSpherePos;
flat in uint   id;

// 3 16 bits for position.
// 3 16 bits for normal.
// 1 32 bits for padding.
layout( location = 0 ) out uvec4 outViewPositionNormal;
// 3 32 bits for color.
// 1 32 bits for specular.
layout( location = 1 ) out vec4 outColor;

layout( location = 2 ) out uint outId;

float computeDepth( const vec3 v )
{
	// Computes 'v' NDC depth ([-1,1])
	const float ndcDepth = ( v.z * uProjMatrix[ 2 ].z + uProjMatrix[ 3 ].z ) / (uIsPerspective ? -v.z : 1.0);
	// Return depth according to depth range
	return ( gl_DepthRange.diff * ndcDepth + gl_DepthRange.near + gl_DepthRange.far ) * 0.5f;
}

void main()
{
	float a, b, delta;

	if (uIsPerspective)
	{
		a = dot( viewImpPos, viewImpPos );
		// b = -dot(viewImpPos, viewSpherePos);
		// But '-' is useless since 'b' is squared for 'delta'.
		// So for 't', '-' is also removed.
		b = dot( viewImpPos, viewSpherePos );
		const float c = dotViewSpherePos - sphereRad * sphereRad;
		delta = b * b - a * c;
	}
	else // Orthographic
	{
		a = 1.0;
		const vec3 OmC = viewImpPos - viewSpherePos;
		b = OmC.z;
		const float c = dot(OmC, OmC) - sphereRad * sphereRad;
		delta = OmC.z * OmC.z - c;
	}

	if ( delta < 0.f )
	{
#ifdef SHOW_IMPOSTORS
		// Show impostors for debugging purpose.
		uvec4 colorNormal = uvec4( 0 );
		// fill G-buffers.
		uvec4 viewPositionNormalCompressed;
		viewPositionNormalCompressed.x = packHalf2x16( viewImpPos.xy );
		viewPositionNormalCompressed.y = packHalf2x16( vec2( viewImpPos.z, -viewSpherePos.x ) );
		viewPositionNormalCompressed.z = packHalf2x16( -viewSpherePos.yz );
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
		// '-' is removed (see 'b').
		const float t = ( b - sqrt( delta ) ) / a;

		// Compute hit point and normal (always in view space).
		const vec3 hit	  = uIsPerspective ? viewImpPos * t : viewImpPos + vec3(0, 0, -t);
		const vec3 normal = normalize( hit - viewSpherePos );

		// Fill depth buffer.
		gl_FragDepth = computeDepth( hit );

		// Compress position and normal.
		uvec4 viewPositionNormalCompressed;
		viewPositionNormalCompressed.x = packHalf2x16( hit.xy );
		viewPositionNormalCompressed.y = packHalf2x16( vec2( hit.z, normal.x ) );
		viewPositionNormalCompressed.z = packHalf2x16( normal.yz );
		viewPositionNormalCompressed.w = 0; // Padding.

		// Output data.
		outViewPositionNormal = viewPositionNormalCompressed;
		outColor			  = vec4( sphereColor, 32.f ); // w = specular shininess.
		outId                 = id;
	}
}

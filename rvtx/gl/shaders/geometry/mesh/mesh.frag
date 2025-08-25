#version 450

//#define SHOW_IMPOSTORS

layout (depth_greater) out float gl_FragDepth;

layout(std140, binding = 0) uniform VertexSettings
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	bool  uIsPerspective;
	bool  uUseSingleId;
};

layout(std430, binding = 2) buffer Ids
{
    uint ids[];
};

in vec3 vViewPos;
in vec3 vNormal;
in vec3 vColor;

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
		// Compute hit point and normal (always in view space).
		const vec3 hit	  = vViewPos;
		const vec3 normal = vNormal;

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
		outColor			  = vec4( vColor, 32.f ); // w = specular shininess.
		outId                 = uUseSingleId ? ids[0] : ids[gl_PrimitiveID];
	
}

#version 450

layout( binding = 0 ) uniform sampler2D depthTexture;

layout( location = 0 ) out float linearDepth;

uniform vec4 uClipInfo;
uniform bool uIsPerspective;

float linearizeDepth( const vec4 clipInfo, const float depth )
{
	if (uIsPerspective)
		return clipInfo[0] / ( clipInfo[1] - depth * clipInfo[2] );
	return depth * clipInfo[2] + clipInfo[3];
}

void main() 
{ 
	linearDepth = linearizeDepth( uClipInfo, texelFetch( depthTexture, ivec2( gl_FragCoord.xy ), 0 ).x ); 
}

#version 450

struct Sphere
{
    vec3  position;
    float radius;
    vec3  color;
    float visibility;
};

layout(std140, binding = 0) uniform SphereSettings
{
	mat4  uMVMatrix;
	mat4  uProjMatrix;
	float uRadiusAdd;
	bool  uIsPerspective;
};

layout(std140, binding = 1) buffer Spheres
{
    Sphere spheres[];
};

layout(std430, binding = 2) buffer SpheresIDs
{
    uint ids[];
};

flat out vec3  vViewSpherePos; // Sphere position in view space.
flat out vec3  vSphereColor;
flat out float vSphereRad;
flat out uint  vVisible;
flat out vec3  vImpU; // Impostor vectors.
flat out vec3  vImpV;
flat out float vDotViewSpherePos;
flat out uint  vId;

void main()
{
	Sphere sphere  = spheres[gl_VertexID];
	vViewSpherePos = vec3( uMVMatrix * vec4( sphere.position, 1. ) );
	vSphereColor   = sphere.color;
	vSphereRad	   = sphere.radius + uRadiusAdd;
	vVisible	   = uint(sphere.visibility);
	vId            = ids[gl_VertexID];

	// Compute normalized view vector.
	vDotViewSpherePos		  = dot( vViewSpherePos, vViewSpherePos );
	const float dSphereCenter = sqrt( vDotViewSpherePos );
	const vec3	view		  = vViewSpherePos / dSphereCenter;

	if (uIsPerspective)
	{
		// Impostor in front of the sphere.
		vec3 viewImpPos = vViewSpherePos - vSphereRad * view;

		// Compute impostor size.
		const float sinAngle = vSphereRad / dSphereCenter;
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
		vImpU = vec3(-1, 0, 0) * vSphereRad;
		vImpV = vec3(0, -1, 0) * vSphereRad; 

		gl_Position = vec4( vViewSpherePos + vec3(0, 0, vSphereRad), 1.f );
	}
}

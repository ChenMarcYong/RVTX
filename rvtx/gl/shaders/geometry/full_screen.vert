#version 450

out vec2 uv;

void main() 
{
	// Reference: https://github.com/SaschaWillems/Vulkan/blob/master/data/shaders/glsl/deferred/deferred.vert
	uv = vec2((gl_VertexID  << 1) & 2, gl_VertexID  & 2);
	gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
}

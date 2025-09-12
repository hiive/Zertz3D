#version 330 core

// Input Attributes
layout(location = 0) in vec4 vtx_position;    // POSITION
layout(location = 1) in vec3 vtx_normal;      // NORMAL
layout(location = 2) in vec2 vtx_texcoord0;   // TEXCOORD0

// Uniforms
uniform vec4 k_time;
uniform vec4 k_wateranim;
uniform mat4 mat_modelproj;
uniform mat4 mat_modelview;
uniform mat4 mat_projection;

// Outputs to Fragment Shader
out vec4 l_texcoord0_out;     // Projective Coordinates for Reflection
out vec2 l_texcoord1_out;     // Texture Coordinates for Distortion

void main()
{
    // Transform vertex position to clip space
    gl_Position = mat_modelproj * vtx_position;

    // Compute projective texture coordinates
    mat4 scaleMatrix = mat4(
        vec4(0.5, 0.0, 0.0, 0.0),
        vec4(0.0, 0.5, 0.0, 0.0),
        vec4(0.0, 0.0, 0.5, 0.0),
        vec4(0.5, 0.5, 0.5, 1.0)
    );

    mat4 matMR = mat_modelproj * scaleMatrix;
    l_texcoord0_out = matMR * vtx_position;

    // Water distortion map: animate and scale distortions
    l_texcoord1_out = vtx_texcoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x;
}
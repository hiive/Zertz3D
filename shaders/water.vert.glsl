#version 330 core

// Input attributes
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
out vec4 l_texcoord0_out;   // TEXCOORD0 (Projective Coordinates for Reflection)
out vec4 l_texcoord1_out;   // TEXCOORD1 (Texture Coordinates for Distortion)

void main()
{
    // Transform vertex position by combined model-projection matrix
    gl_Position = mat_modelproj * vtx_position;

    // Scale matrix for projective texture coordinates
    mat4 scaleMatrix = mat4(
        0.5, 0.0, 0.0, 0.5,
        0.0, 0.5, 0.0, 0.5,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0
    );

    // Compute projective matrix (Scale * Model-Projection)
    mat4 matMR = scaleMatrix * mat_modelproj;

    // Transform the vertex position by the projective texture matrix
    l_texcoord0_out = matMR * vtx_position;

    // Water distortion map: animate and scale distortions
    l_texcoord1_out.xy = vtx_texcoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x;
    l_texcoord1_out.zw = vec2(1.0, 1.0); // Ensure w component is 1.0 for vec4
}
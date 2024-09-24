#version 330 core

// Input attributes
layout(location = 0) in vec4 vtx_position;    // POSITION
layout(location = 1) in vec3 vtx_normal;      // NORMAL
layout(location = 2) in vec2 vtx_texcoord0;   // TEXCOORD0
layout(location = 3) in vec2 vtx_texcoord3;   // TEXCOORD3

// Uniforms
uniform mat4 mat_modelproj;
uniform mat4 trans_model_to_world;
uniform vec4 k_lightvec;
uniform vec4 k_lightcolor;
uniform vec4 k_ambientlight;
uniform vec4 k_tscale;

// Outputs to Fragment Shader
out vec4 l_brightness;
out vec4 l_mpos;
out vec2 l_texcoord0_out;
out vec2 l_texcoord3_out;

void main()
{
    // Calculate world-space position
    l_mpos = trans_model_to_world * vtx_position;

    // Transform to clip space
    gl_Position = mat_modelproj * vtx_position;

    // Scale texture coordinates
    l_texcoord0_out = vtx_texcoord0 * k_tscale.xy;
    l_texcoord3_out = vtx_texcoord3;

    // Lighting calculations
    vec3 N = normalize(vtx_normal);
    vec3 L = normalize(k_lightvec.xyz);
    float diffuse = max(dot(-N, L), 0.0);
    l_brightness = vec4(diffuse * k_lightcolor.rgb + k_ambientlight.rgb, 1.0);
}
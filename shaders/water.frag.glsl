#version 330 core

// Inputs from Vertex Shader
in vec4 l_texcoord0_out;   // Projective Coordinates for Reflection
in vec2 l_texcoord1_out;   // Texture Coordinates for Distortion

// Uniforms
uniform sampler2D tex_0;            // Reflection Texture
uniform sampler2D tex_1;            // Distortion Texture
uniform vec4 k_waterdistort;        // Water Distortion Parameters

// Output color
out vec4 o_color;

void main()
{
    // Sample the distortion texture
    vec2 distortionSample = texture(tex_1, l_texcoord1_out).xy;

    // Calculate distortion
    vec2 distortion = (distortionSample - vec2(k_waterdistort.x)) * k_waterdistort.y;

    // Adjust projective coordinates with distortion
    vec4 projCoord = l_texcoord0_out;
    projCoord.xy += distortion * projCoord.w;

    // Sample the reflection texture
    vec4 reflection = textureProj(tex_0, projCoord);

    // Refraction factor
    float factor = k_waterdistort.z;

    // Refraction color
    float ref = k_waterdistort.w;
    vec4 refraction = vec4(ref, ref, ref, 1.0);

    // Blend reflection and refraction
    o_color = mix(reflection, refraction, factor);

    // Set alpha for transparency
    o_color.a = 0.6;
}
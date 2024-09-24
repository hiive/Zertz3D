#version 330 core

// Inputs from Vertex Shader
in vec4 l_texcoord0_out;   // TEXCOORD0 (Projective Coordinates for Reflection)
in vec4 l_texcoord1_out;   // TEXCOORD1 (Texture Coordinates for Distortion)

// Uniforms
uniform sampler2D tex_0;            // Reflection Texture
uniform sampler2D tex_1;            // Distortion Texture
uniform vec4 k_waterdistort;        // Water Distortion Parameters

// Output color
out vec4 o_color;

void main()
{
    // Calculate distortion from distortion map
    // Assuming distortion is in the xy channels
    vec2 distortionSample = texture(tex_1, l_texcoord1_out.xy).xy;
    vec2 distortion = normalize(distortionSample - vec2(k_waterdistort.x)) * k_waterdistort.y;

    // Projectively sample the reflection texture with distorted coordinates
    // Ensure that l_texcoord0_out has a valid w component
    vec4 projCoord = l_texcoord0_out + vec4(distortion, 0.0, 0.0);
    vec4 reflection = textureProj(tex_0, projCoord);

    // Refraction factor: smaller numbers make the water appear more reflective ("shinier")
    float factor = k_waterdistort.z;

    // Refraction color: (1.0 = perfect mirror, 0.0 = total refraction)
    float ref = k_waterdistort.w;
    vec4 refraction = vec4(ref, ref, ref, 0.0);

    // Blend reflection and refraction based on the factor
    o_color = mix(reflection, refraction, factor);
    o_color = vec4(1.0, 0.0, 0.0, 1.0);
    // Optionally set alpha component to transparency
    o_color.a = 0.6;  // Adjust alpha as needed
}
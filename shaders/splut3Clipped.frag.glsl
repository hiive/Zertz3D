#version 330 core

// Inputs from Vertex Shader
in vec4 l_brightness;
in vec4 l_mpos;
in vec2 l_texcoord0_out;
in vec2 l_texcoord3_out;

// Uniforms
uniform vec4 k_waterlevel;
uniform sampler2D tex_0;
uniform sampler2D tex_1;
uniform sampler2D tex_2;
uniform sampler2D tex_3;
uniform sampler2D tex_4;
uniform sampler2D tex_5;

// Output color
out vec4 o_color;

void main()
{
    // Clipping based on water level
    if (l_mpos.z < k_waterlevel.z)
        discard;

    // Sample textures
    vec4 tex1 = texture(tex_0, l_texcoord0_out);
    vec4 tex2 = texture(tex_1, l_texcoord0_out);
    vec4 tex3 = texture(tex_2, l_texcoord0_out);

    // Sample alpha maps (assuming alpha is in the Z component)
    float alpha1 = texture(tex_3, l_texcoord3_out).z;
    float alpha2 = texture(tex_4, l_texcoord3_out).z;
    float alpha3 = texture(tex_5, l_texcoord3_out).z;

    // Combine textures with their respective alpha values
    vec4 combinedColor = tex1 * alpha1;
    combinedColor += tex2 * alpha2;
    combinedColor += tex3 * alpha3;

    // Apply lighting
    combinedColor *= l_brightness;

    // Set alpha to 1.0
    combinedColor.a = 1.0;

    // Output the final color
    o_color = combinedColor;
}
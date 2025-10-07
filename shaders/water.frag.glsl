#version 330 core

// in vec2 l_texcoord0;
in vec2 l_texcoord1;

uniform sampler2D tex_0;
uniform sampler2D tex_1;
uniform vec4 k_waterdistort;
uniform vec2 k_screen_size;

out vec4 p3d_FragColor;

void main()
{
    // Use screen coordinates to sample the reflection texture
    vec2 screenUV = gl_FragCoord.xy / k_screen_size;

    // Sample reflection texture directly (buffer now matches screen aspect ratio)
    vec4 reflection = texture(tex_0, screenUV);

    // Add distortion
    vec2 distortion = (texture(tex_1, l_texcoord1).xy - k_waterdistort.x) * k_waterdistort.y;
    vec4 distorted = texture(tex_0, screenUV + distortion);

    // Blend
    p3d_FragColor = mix(distorted, vec4(k_waterdistort.w), k_waterdistort.z);
    p3d_FragColor.a = 0.8;

    // Force depth to render behind board but in front of skybox
    gl_FragDepth = 0.98;
}
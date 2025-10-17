#version 330 core

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

uniform vec4 k_time;
uniform vec4 k_wateranim;
uniform mat4 p3d_ModelViewProjectionMatrix;

// out vec2 l_texcoord0;
out vec2 l_texcoord1;

void main()
{
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;

    // Just use the built-in texture coordinates for now
    // l_texcoord0 = p3d_MultiTexCoord0;
    l_texcoord1 = p3d_MultiTexCoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x;
}
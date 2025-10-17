#version 330 core

uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform float outline_thickness;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;

out vec3 v_view_normal;
out vec3 v_view_pos;

void main() {
    vec3 view_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    vec4 view_vertex = p3d_ModelViewMatrix * p3d_Vertex;
    v_view_normal = view_normal;
    v_view_pos = view_vertex.xyz;
    view_vertex.xyz += view_normal * outline_thickness;
    gl_Position = p3d_ProjectionMatrix * view_vertex;
}

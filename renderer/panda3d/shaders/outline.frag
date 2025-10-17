#version 330 core

uniform vec4 outline_color;
uniform float outline_threshold;
uniform float outline_dilate;

in vec3 v_view_normal;
in vec3 v_view_pos;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(v_view_normal);
    vec3 view_dir = normalize(-v_view_pos);
    float ndotv = abs(dot(normal, view_dir));
    float expand = outline_threshold + outline_dilate * fwidth(ndotv);
    if (ndotv < expand) {
        float denom = max(expand - outline_threshold, 1e-5);
        float fade = clamp((expand - ndotv) / denom, 0.0, 1.0);
        // float fade = 0.5;
        fragColor = vec4(outline_color.rgb, outline_color.a  * 0.25);
        // fragColor = vec4(outline_color.rgb, outline_color.a * 0.1);
    } else {
        discard;
    }
}

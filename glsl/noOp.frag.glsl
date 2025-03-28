#version 330 core

uniform sampler2D u_Texture;
uniform vec2 u_ScreenDims;
uniform int u_Iterations;

in vec3 fs_Pos;
in vec2 fs_UV;

out vec4 out_Col;
void main()
{
    vec4 color = texture(u_Texture, fs_UV);
    // Apply the Reinhard operator and gamma correction before outputting color
    // You must take your render, which has its colors stored as high dynamic range RGB values,
    // and convert it to standard RGB range by first applying the Reinhard operator to its colors then gamma correcting them

    vec3 reinhardMap = color.rgb / (color.rgb + vec3(1.f)); // L = L / 1 + L

    out_Col = vec4(pow(reinhardMap, vec3(1.f / 2.2f)), color.a);        // gamma correcting = color ^ 1/g
}

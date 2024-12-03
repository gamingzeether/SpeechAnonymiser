#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    vec4 frequencies[512];
} ubo;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    int index = gl_VertexIndex;
    float amplitude = ubo.frequencies[index / 4][index % 4];
    gl_Position = vec4(inPosition.x, amplitude - 0.9, 0.0, 1.0);
    fragColor = vec3(1.0, 1.0, 1.0);
}
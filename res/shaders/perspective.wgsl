// Vertex shader

struct InstanceInput {
    [[location(5)]] model_matrix_0: vec4<f32>;
    [[location(6)]] model_matrix_1: vec4<f32>;
    [[location(7)]] model_matrix_2: vec4<f32>;
    [[location(8)]] model_matrix_3: vec4<f32>;
};

struct CameraUniform {
    pers_view_proj: mat4x4<f32>;
    ortho_view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
    [[location(1)]] position: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    out.clip_position = camera.pers_view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.position = vec2<f32>(model.position[0], model.position[1]);
    out.tex_coords = model.tex_coords;
    return out;
}


// Fragment shader

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let from_tex = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let from_pos = vec4<f32>(0.5, in.position, 0.0);
    if (from_tex[3] == 0.0) {
        return from_pos;
    }
    return from_tex;
}

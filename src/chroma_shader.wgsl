// Vertex shader

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] position: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

// Fragment shader

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(0.5, in.position, 0.0);
}

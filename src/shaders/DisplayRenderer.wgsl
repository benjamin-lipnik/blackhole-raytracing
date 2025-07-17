
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct VertexOutput {
	@builtin(position) pos : vec4f,
	@location(0) uv : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1, -1),
    vec2f( 3, -1),
    vec2f(-1,  3),
  );
  var output : VertexOutput;
  output.pos = vec4f(pos[i], 0.0, 1.0);
  output.uv =  (vec2f(pos[i].x,-pos[i].y) + vec2f(1,1)) * 0.5;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(tex, tex_sampler, input.uv);
}

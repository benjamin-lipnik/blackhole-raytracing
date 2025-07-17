@group(0) @binding(0) var<uniform> un: Uniforms;
@group(0) @binding(1) var skybox_sampler: sampler;
@group(0) @binding(2) var skybox_texture: texture_cube<f32>;
@group(0) @binding(3) var blackbody_color: texture_2d<f32>;

struct Uniforms {
	clip_unproject_dir: mat4x4f,
	camera_pos: vec3f,

    g_min_step: f32,
    g_rs: f32,
    g_temperature: f32,
    g_disk_inner: f32,
    g_disk_outer: f32,
};

const disc_normal = vec3f(0,0,1);

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) pos: vec4f,
};

@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOutput {
    let pos = array(
        vec2f(-1, 3),
        vec2f(-1,-1),
        vec2f( 3,-1)
    );
    var vsOut: VSOutput;
    vsOut.position = vec4f(pos[i], 1, 1);
    vsOut.pos = vsOut.position;
    return vsOut;
}

// direction must be normalized
// returns distance to closest intersection. Or a negative value if there is not intersection.
fn intersect_sphere(ray_origin: vec3f, ray_dir: vec3f, sphere_center: vec3f, rs: f32) -> f32 {
	let oc = ray_origin - sphere_center;
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - rs * rs;
    let discriminant = b * b - 4.0 * c;

    if (discriminant < 0.0) {
        return -1.0; // No intersection
    }

    // Return nearest intersection distance
    return (-b - sqrt(discriminant)) / (2.0);
}

fn intersect_ring_disc(ray_origin: vec3f, ray_dir: vec3f, disc_center: vec3f, disc_normal: vec3f, inner_radius: f32, outer_radius: f32) -> f32 {
    let epsilon = 1e-5;

    let denom = dot(ray_dir, disc_normal);
    if abs(denom) < epsilon {
        return -1.0; // Ray is parallel to disc
    }

    let to_center = disc_center - ray_origin;
    let t = dot(to_center, disc_normal) / denom;
    if t < 0.0 {
        return -1.0; // Intersection behind origin
    }

    let hit_point = ray_origin + ray_dir * t;
    let radial_vec = hit_point - disc_center;
    let dist_squared = dot(radial_vec, radial_vec);

    let inner2 = inner_radius * inner_radius;
    let outer2 = outer_radius * outer_radius;

    if dist_squared < inner2 || dist_squared > outer2 {
        return -1.0; // Outside ring bounds
    }

    return t; // Valid ring hit
}


@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {

    // Use this so it doesn't get optimized out.
    let tmp = textureSample(blackbody_color, skybox_sampler, vec2f(0,0));

    var dir_h = un.clip_unproject_dir * vsOut.pos;
	var dir = normalize(dir_h.xyz / dir_h.w);
    var  origin = un.camera_pos;

    var color = textureSample(skybox_texture, skybox_sampler, vec3f(dir.x, dir.z, -dir.y));

    var intersected = false;
    var min_dist : f32 = 10000000;

    // Check sphere intersection
    {
		let d = intersect_sphere(origin, dir, vec3f(0,0,0), un.g_rs);
		if(d >= 0) {
            intersected = true;
            if(d < min_dist) {
                min_dist = d;
		        color = vec4f(0,0,0,1);
            }
		}
    }

    // Check disc intersection
    {
		let d = intersect_ring_disc(origin, dir, vec3f(0,0,0), disc_normal, un.g_disk_inner, un.g_disk_outer);
		if(d >= 0) {
            intersected = true;
            if(d < min_dist) {
                min_dist = d;
		        color = vec4f(1,1,1,1);
		    }
		}
    }
    return color;
}

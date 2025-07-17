@group(0) @binding(0) var<uniform> un: Uniforms;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var skybox_texture: texture_cube<f32>;
@group(0) @binding(3) var blackbody_color: texture_2d<f32>;

const g_escape_dist : f32 = 500.0;
const PI: f32 = 3.14159265;

struct Uniforms {
	clip_unproject_dir: mat4x4f,
	camera_pos: vec3f,

    g_min_step: f32,
    g_rs: f32,
    g_temperature: f32,
    g_disk_inner: f32,
    g_disk_outer: f32,
    g_falloff_rate: f32,
    g_beam_exponent: f32,
    g_redshift: f32,
};

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) pos: vec4f,
};

@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOutput {
    let pos = array( vec2f(-1, 3), vec2f(-1,-1), vec2f( 3,-1));
    var vsOut: VSOutput;
    vsOut.position = vec4f(pos[i], 1, 1);
    vsOut.pos = vsOut.position;
    return vsOut;
}

@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
    let dir_h = un.clip_unproject_dir * vsOut.pos;
	let dir = normalize(dir_h.xyz / dir_h.w);
    let origin = un.camera_pos;
    let pixel_xy = vsOut.position.xy;

    return raymarch(origin, dir, pixel_xy);
}

fn float_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Color blending helper functions

// Premultiplied alpha blending
fn blend_over_pm(front: vec4f, back: vec4f) -> vec4f {
    return front + (1.0 - front.a) * back;
}

// straight to premultiplied alpha
fn color2pm(straight: vec4f) -> vec4f {
    return vec4f(straight.rgb * straight.a, straight.a);
}

// premultiplied to straight alpha
fn color2straight(pm: vec4f) -> vec4f {
    return(select(vec4f(pm.rgb / pm.a, pm.a), vec4f(0), pm.a <= 0.0));
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Noise helper functions

fn hash(p: vec2<i32>) -> f32 {
    let x = u32(p.x);
    let y = u32(p.y);
    var h = x * 374761393u + y * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return f32(h & 0x7fffffffu) / f32(0x7fffffffu);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);

    let a = hash(i);
    let b = hash(i + vec2<i32>(1, 0));
    let c = hash(i + vec2<i32>(0, 1));
    let d = hash(i + vec2<i32>(1, 1));

    let u = vec2<f32>(fade(f.x), fade(f.y));
    let ab = mix(a, b, u.x);
    let cd = mix(c, d, u.x);
    return   mix(ab, cd, u.y);
}

fn fbm2d(p: vec2<f32>, octaves: i32, gain: f32, lacunarity: f32) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i = i + 1) {
        total = total + noise2d(pos * frequency) * amplitude;
        amplitude = amplitude * gain;
        frequency = frequency * lacunarity;
    }

    return total;
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Coordinate systems helper functions

fn cartesian2spherical_jacobian(pos_cart: vec3<f32>) -> mat3x3<f32> {
    let x = pos_cart.x;
    let y = pos_cart.y;
    let z = pos_cart.z;

    let r = sqrt(x * x + y * y + z * z);
    var rho2 = x * x + y * y;
    var rho = sqrt(rho2);

    // Avoid division by zero
    if (rho2 < 1e-8) {
        rho2 = 1e-8;
        rho = 1e-4;
    }

    // Column-major order
    return mat3x3<f32>(
        x/r,    (x * z) / (r * r * rho),    -y / rho2,
        y/r,    (y * z) / (r * r * rho),     x / rho2,
        z/r,    -rho / (r * r),              0.0
    );
}

// Converts a Cartesian point to spherical coordinates (r, theta, phi)
fn point_cartesian2spherical(pos_cart: vec3f) -> vec3f {
    let x = pos_cart.x;
    let y = pos_cart.y;
    let z = pos_cart.z;

    let r = length(pos_cart);
    let theta = acos(z / r);
    let phi = atan2(y, x) + PI;

    return vec3f(r, theta, phi);
}

// Converts spherical coordinates (r, theta, phi) to Cartesian (x, y, z)
fn point_spherical2cartesian(pos_sph: vec3f) -> vec3f {
    let r     = pos_sph.x;
    let theta = pos_sph.y;
    let phi   = pos_sph.z - PI;

    let sin_theta = sin(theta);
    let x = r * sin_theta * cos(phi);
    let y = r * sin_theta * sin(phi);
    let z = r * cos(theta);

    return vec3f( x, y, z );
}

// Converts a Cartesian vector to spherical components at a given Cartesian position
fn vector_cartesian2spherical(pos_cart: vec3f, dir_cart: vec3f) -> vec3f {
    let jacobian = cartesian2spherical_jacobian(pos_cart);
    return jacobian * dir_cart;
}

fn vector_spherical2cartesian(sph_origin: vec3f, sph_dir: vec3f) -> vec3f {
	let cart_origin = point_spherical2cartesian(sph_origin);
	return vector_spherical2cartesian2(cart_origin, sph_dir);
}

fn vector_spherical2cartesian2(pos_cart: vec3<f32>, dir_sph: vec3<f32>) -> vec3<f32> {
    let x = pos_cart.x;
    let y = pos_cart.y;
    let z = pos_cart.z;

    let r = sqrt(x * x + y * y + z * z);
    var rho2 = x * x + y * y;
    var rho = sqrt(rho2);

    if (rho2 < 1e-8) {
        rho2 = 1e-8;
        rho = 1e-4;
    }

	let jacobian = mat3x3f(
		x/r,           y/r,          z/r,
		(z*x)/rho,     (z*y)/rho,    -rho,
		-y,            x,            0
	);
    return jacobian * dir_sph;
}

// B versions (same coordinate system, rotated 90 deg around the x axis)

fn point_sphericalB2cartesian(pos_sph: vec3f) -> vec3f {
    let cart_B = point_spherical2cartesian(pos_sph);
    return vec3f(cart_B.x, -cart_B.z, cart_B.y);
}

fn point_cartesian2sphericalB(pos_cart: vec3f) -> vec3f {
    let pos_cartB = vec3f(pos_cart.x, pos_cart.z, -pos_cart.y);
    return point_cartesian2spherical(pos_cartB);
}

fn vector_cartesian2sphericalB(pos_cart: vec3f, dir_cart: vec3f) -> vec3f {
    let pos_cartB = vec3f(pos_cart.x, pos_cart.z, -pos_cart.y);
    let dir_cartB = vec3f(dir_cart.x, dir_cart.z, -dir_cart.y);
    return vector_cartesian2spherical(pos_cartB, dir_cartB);
}

fn vector_sphericalB2cartesian2(pos_cart: vec3f, sph_dir: vec3f) -> vec3f {
    let pos_cartB = vec3f(pos_cart.x, pos_cart.z, -pos_cart.y);
    let t = vector_spherical2cartesian2(pos_cartB, sph_dir);
    return vec3f(t.x, -t.z, t.y);
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Schwarzchild metric helper functions

fn lower_vector(origin: vec4f, vector: vec4f) -> vec4f  {
    let x = origin;
    let p = vector;

    let r = x.x;
	let rs = un.g_rs;
	let A = 1.0 - (rs / r);
	let s_th = sin(x.y);

	return vec4f(
        p.x * (1.0 / A),
        p.y * (r*r),
        p.z * (r*r*s_th*s_th),
        p.w * (-A)
    );
}

fn calc_schwarzchild_spatial_norm_sq(origin: vec3f, vec: vec3f) -> f32 {
	let r = origin.x;
	let rs = un.g_rs;
	let A = 1.0 - (rs / r);
	let s_th = sin(origin.y);

	let lowered_vec = vec3f(
		vec.x * (1.0 / A),
		vec.y * (r*r),
		vec.z * (r*r*s_th*s_th)
	);
	return dot(lowered_vec, vec);
}

// Given Schwarzschild radius, position, and spatial components of a null vector,
// compute the required time component v^t so that g(v, v) = 0.
fn determine_schwarzschild_time(origin: vec4f, vec: vec4f) -> f32 {
    let rs = un.g_rs;
    let r = origin.x;
    let theta = origin.y;
    let f = 1.0 - (rs / r);

    let vr     = vec.x;
    let vtheta = vec.y;
    let vphi   = vec.z;

    let sin_theta = sin(theta);
    let vt_squared =
        (vr * vr) / (f * f)
        + (r * r / f) * (vtheta * vtheta + sin_theta * sin_theta * vphi * vphi);

    return sqrt(vt_squared);
}

fn geodesic_second_derivative(x: vec4f, p: vec4f, dx: ptr<function, vec4f>, dp: ptr<function, vec4f>) {
	*dx = p;

	let rs = un.g_rs;
	let r = x.x;
	let th = x.y;
	var drs = r-rs;
	if (drs < 1e-6) {
        drs = 1e-6;
    }
	let s_th = sin(th);
	let c_th = cos(th);
	let cot_th = c_th/s_th;

	let p_r = p.x;
	let p_th = p.y;
	let p_fi = p.z;
	let p_t = p.w;

	*dp = vec4f(
		(-rs*drs*p_t*p_t)/(2*r*r*r) + (rs*p_r*p_r)/(2*r*drs) + drs*p_th*p_th + drs*s_th*s_th*p_fi*p_fi,
		(-2*p_r*p_th)/(r) + s_th*c_th*p_fi*p_fi,
		-2*((p_r*p_fi)/(r) + cot_th*p_th*p_fi),
		(-rs*p_t*p_r)/(r*drs)
	);
    return;
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Raymarching logic

fn raymarch(_pos_cart: vec3f, _dir_cart: vec3f, pixel_xy: vec2f) -> vec4f {
    var pos_cart = _pos_cart;
    var dir_cart = _dir_cart;

    var color_pm = vec4f(0,0,0,0);

    // Convert coordinates to spherical -> schwarzchild.
    var coordinate_systemB = abs(cos(point_cartesian2spherical(pos_cart).y)) >= 0.866;
    var x = vec4f(select(
        point_cartesian2spherical( pos_cart),
        point_cartesian2sphericalB(pos_cart),
        coordinate_systemB), 0.0);

    var p = vec4f(select(
        vector_cartesian2spherical( pos_cart, dir_cart),
        vector_cartesian2sphericalB(pos_cart, dir_cart),
        coordinate_systemB), 0.0);

	p.x *= sqrt(1.0 - un.g_rs/x.x);
	p.w = determine_schwarzschild_time(x, p);

    var last_disk_crossing: i32 = -1000;
    let max_steps = i32(g_escape_dist / un.g_min_step);

    for(var i: i32 = 0; i < max_steps; i = i+1) {
        if(color_pm.a >= 0.99) {
            break;
        }
		let old_x = x;
		let old_p = p;
        var dynamic_step_size = 1.0;

        // RK4
        {
            var dx: vec4f;
            var dp: vec4f;

            var k1x: vec4f;
            var k1p: vec4f;
            geodesic_second_derivative(x, p, &k1x, &k1p);

		    // Dynamic step size
		    let dp_norm_sq = calc_schwarzchild_spatial_norm_sq(x.xyz, dp.xyz) * 5000.0;
		    let m1 = 1.0 / clamp(dp_norm_sq, 0.1, 1.0);
            let m2 = clamp(x.x*0.05, 0.001, 1000.0);
            let step_mul = m1*m2;
            dynamic_step_size = un.g_min_step * step_mul;

            geodesic_second_derivative(
                x + 0.5 * dynamic_step_size * k1x,
                p + 0.5 * dynamic_step_size * k1p,
                &dx, &dp
            );
            let k2x = dx;
            let k2p = dp;

            geodesic_second_derivative(
                x + 0.5 * dynamic_step_size * k2x,
                p + 0.5 * dynamic_step_size * k2p,
                &dx, &dp
            );
            let k3x = dx;
            let k3p = dp;

            geodesic_second_derivative(
                x + dynamic_step_size * k3x,
                p + dynamic_step_size * k3p,
                &dx, &dp
            );
            let k4x = dx;
            let k4p = dp;

            x += dynamic_step_size * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
            p += dynamic_step_size * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;

            x.y = float_mod(x.y, PI);
            x.z = float_mod(x.z, 2.0*PI);
        }

        // BH detection
        let singularity = max(length(x - old_x), length(p - old_p)) > (10000*dynamic_step_size);
        if(singularity || x.x <= 1.0*un.g_rs) {
            color_pm = blend_over_pm(color_pm, vec4f(0,0,0,1));
            break;
        }

		// Disc detection
		if((i - last_disk_crossing) > 5){ // This helps remove false double disk detections after coordinate system swaps
			// Disk detection (A)
			let old_c_th = cos(old_x.y);
			let new_c_th = cos(x.y);
			let a = (old_c_th * new_c_th) <= 0.0;

            // Disk detection (B)
			let old_s_fi = sin(old_x.z);
			let new_s_fi = sin(x.z);
			let b = (old_s_fi * new_s_fi) <= 0.0;

            if(select(a,b,coordinate_systemB)) {
                last_disk_crossing = i;

                // Convert into A (in A, the accretion disk is at @ theta = pi/2)
                var ax: vec4f;
                var px: vec4f;
                var ax_cart: vec3f;
                var px_cart: vec3f;

                let spherical_p3 = vec3(p.x / sqrt(1.0 - un.g_rs/x.x), p.yz);
                if(coordinate_systemB) {
                    ax_cart =  point_sphericalB2cartesian(x.xyz);
                    px_cart = vector_sphericalB2cartesian2(ax_cart, spherical_p3);
                    ax = vec4f( point_cartesian2spherical(ax_cart), x.w);
                    px = vec4f(vector_cartesian2spherical(ax_cart, px_cart), p.w);
                    px.x *= sqrt(1.0 - un.g_rs/x.x);
                }else {
                    ax = x;
                    px = p;
                    ax_cart =  point_spherical2cartesian(x.xyz);
                    px_cart = vector_spherical2cartesian2(ax_cart, spherical_p3);
                }


                // Fix overstep, by positionin x directly onto disk intersection
                let overstep : f32 = abs(ax_cart.z / px_cart.z);
                ax_cart = ax_cart - overstep*px_cart;
                ax = vec4f(point_cartesian2spherical(ax_cart), ax.w - overstep*px.w);

                var cc = computeDiskColor(ax, px, ax_cart, px_cart);
                color_pm = blend_over_pm(color_pm, color2pm(cc));
			}

		}

	    // Escape radius
        if(x.x > (un.g_rs * g_escape_dist)) {
            break;
		}

        // Pole detection
		{
			if(abs(cos(x.y)) >= 0.866) {  // 30 deg around poles.
				// Convert to vec3
				var x3 = x.xyz;
				var p3 = p.xyz;

				if(!coordinate_systemB) { // Convert into B
					let pos_cart = point_spherical2cartesian(x3);
					var dir_cart = vector_spherical2cartesian2(pos_cart, p3);
					x3 = point_cartesian2sphericalB(pos_cart);
					p3 = vector_cartesian2sphericalB(pos_cart,dir_cart);
				}else { // Convert into A
					let pos_cart = point_sphericalB2cartesian(x3);
					var dir_cart = vector_sphericalB2cartesian2(pos_cart, p3);
					x3 = point_cartesian2spherical(pos_cart);
					p3 = vector_cartesian2spherical(pos_cart,dir_cart);
				}

				coordinate_systemB = !coordinate_systemB;
				x = vec4(x3, x.w);
				p = vec4(p3, p.w);
			}
		}
    }

    // Conversion back to cartesian
	p.x /= sqrt(1.0 - un.g_rs/x.x);
	if(!coordinate_systemB) { // A
        pos_cart =  point_spherical2cartesian(x.xyz);
        dir_cart = vector_spherical2cartesian2(pos_cart, p.xyz);
	}else { // B
        pos_cart =  point_sphericalB2cartesian(x.xyz);
        dir_cart = vector_sphericalB2cartesian2(pos_cart, p.xyz);
    }

    let sky_dir = vec3f(dir_cart.x, dir_cart.z, -dir_cart.y);
    let sky: vec4f = color2pm(textureSampleLevel(skybox_texture, linear_sampler, sky_dir, 0.0));
    let background = sky;
    // let background: vec4f = vec4f(0,0,0,1);
    let out_color = color2straight(blend_over_pm(color_pm, background));
    return vec4(out_color.rgb, 1.0);
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Acretion disk color logic

fn sampleBlackBodyColor(uv: vec2f) -> vec3f {
    let clamped_uv = clamp(vec2f(uv.x, 1.0-uv.y), vec2f(0.0), vec2f(1.0));
    let texel_coords = clamped_uv * 255.0;
    return textureLoad(blackbody_color, vec2u(texel_coords), 0u).rgb;
}

fn computeRedshiftFactor(x: vec4f, p: vec4<f32>) -> f32 {
    let rs = un.g_rs;
    let r = x.x;
    let u_em = vec4f(0,0,sqrt(rs/(r*r*(2*r-3*rs))), 1.0 / sqrt(1.0 - (3*rs)/(2*r)));
    let u_obs = vec4f(0,0,0,1);

    let p_lowered = lower_vector(x,p);
    return max(dot(p_lowered, u_obs) / dot(p_lowered, u_em), 0);
}

fn computeDiskColor(x: vec4f, p: vec4f, x3_cart: vec3f, p3_cart: vec3f) -> vec4f {
    // Use cylindrical coordinates around the disk
    let fi = x.z / (2*PI);
    var uv = vec2f(
        (x.x - un.g_disk_inner) / (un.g_disk_outer - un.g_disk_inner),
        0.5 - abs(fi - 0.5) // Mirror the texture, to remove the seam
    );

    // Sample noise texture
    var surface_texture = fbm2d(vec2f(uv.x * x.x, uv.y)*10, 10, 1.0, 3.0)*0.5;

    // Reduce intensity over distance (falloff)
    let opacity = select(-log(uv.x), exp(un.g_falloff_rate * uv.x * un.g_disk_outer / un.g_rs), uv.x < 0.0);

    // Calculate doppler redshift
    // let redshift = computeRedshiftFactor(x,p);
    let redshift = computeRedshiftFactor(x,p);

    // Relativistic beaming
    surface_texture *= pow(abs(redshift), un.g_beam_exponent);

    // Calculate temperature
    let temperature = un.g_temperature * pow(abs(un.g_disk_inner / x.x), 0.75);

    // Sample blackbody texture
    uv = vec2f(
        (redshift*un.g_redshift - 0.5) / (2.0 - 0.5),
        temperature
    );
    let color = sampleBlackBodyColor(uv);

    // Weight by noise strength and multiplier
    return vec4f(surface_texture * color.xyz, clamp(opacity*3,0,1));
}

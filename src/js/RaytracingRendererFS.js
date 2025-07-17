import * as Util from "./util.js";
import { mat4, vec3 } from "./lib/glMatrix.js";
import * as Scene from "./scene.js";

export default class RaytracingRenderer {
    constructor(shader_path, preview = false) {
        this.fps = 0;
        this.shader_path = shader_path;
        this.preview = preview;
    }
    async initRenderer(config) {
		this.pipeline = await this.initPipeline(config);

        this.linear_sampler = config.device.createSampler({
            magFilter:    'linear',
            minFilter:    'linear',
            mipmapFilter: 'linear',
        });
		this.setResolution(config);
    }
    async initPipeline(config) {
        let shader_code = await Util.loadFileString(this.shader_path);
        const shader_module = config.device.createShaderModule({
            code: shader_code
        });

        return await config.device.createRenderPipelineAsync({
            label: "RaytracingRenderer skybox pipeline",
            layout: "auto",
            vertex: {
                module: shader_module
            },
            fragment: {
                module: shader_module,
                targets: [{ format: "rgba8unorm" }],
            },
        });
	}

    setResolution(config) {
        if(!this.pipeline) {
			console.error("Pipeline not yet inited.");
            return;
		}

        let width  = Util.resolutionRound(config.width);
        let height = Util.resolutionRound(config.height);

        if(this.preview) {
            let preview_resolution = 150*150;
            let aspect = (config.width / config.height);
            width = Util.resolutionRound(Math.sqrt(aspect * preview_resolution));
            height = Util.resolutionRound(width / aspect);
        }

		this.output_texture = config.device.createTexture({
			size: [width, height],
			format: "rgba8unorm",
			usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
		});
        this.output_texture_view = this.output_texture.createView();

	    this.uniform_buffer = config.device.createBuffer({
	    	label: "Raytracing renderer uniform buffer",
        	size: 128 + 8*4,
        	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  	    });

		// Create a new output texture and its bind group.
		// TODO: add view and inverse projection matrix.

        this.bind_group = config.device.createBindGroup({
            label: "Compute shader bind group",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
                { binding: 0, resource: { buffer: this.uniform_buffer } },
                { binding: 1, resource: this.linear_sampler },
                { binding: 2, resource: Scene.skybox_texture_view },
                { binding: 3, resource: Scene.blackbody_color_texture_view },
			]
		});

        return this.output_texture;
    }
    async render(config) {
        let aspect_ratio = config.width / config.height;
	    let non_translated_view = mat4.clone(Scene.camera.getInverseNodeMatrix());
		non_translated_view[12] = 0;
		non_translated_view[13] = 0;
		non_translated_view[14] = 0;

        // Calculating this and copying it into the uniform buffer every frame is kind of wastefull.
        // This could be done once at setResolution
        let p = mat4.perspective(mat4.create(), config.camera_fov, aspect_ratio, 0.1, 100);
		Util.toWebGPU_NDC(p);

		let vp_inverse = mat4.multiply(mat4.create(), p, non_translated_view);
		mat4.invert(vp_inverse, vp_inverse);

		// TODO: Tule kopiraj matrike v uniform buffer.
		config.device.queue.writeBuffer(
        	this.uniform_buffer,
            0,
        	vp_inverse.buffer
        );
		config.device.queue.writeBuffer(
        	this.uniform_buffer,
            64,
			Scene.camera.position.buffer
        );
        config.device.queue.writeBuffer(
            this.uniform_buffer,
            76,
            new Float32Array([
                config.uniforms.g_min_step,
                config.uniforms.g_rs,
                config.uniforms.g_temperature,
                config.uniforms.g_disk_inner_multiplier * config.uniforms.g_rs,
                config.uniforms.g_disk_outer_multiplier * config.uniforms.g_rs,
                config.uniforms.g_falloff_rate,
                config.uniforms.g_beam_exponent,
                config.uniforms.g_redshift,
            ])
        );

        const t0 = performance.now();
        // === RENDER PASS ===
        const renderEncoder = config.device.createCommandEncoder();
        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.output_texture_view,
                    clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
        };

        const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor);
        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bind_group);
        renderPass.draw(3);
        renderPass.end();
        config.device.queue.submit([renderEncoder.finish()]);

        await config.device.queue.onSubmittedWorkDone();
        // const t1 = performance.now();
        // this.fps = 0.95*this.fps + 0.05 * (1000/(t1-t0));
        // console.log("Fragment, FPS:", this.fps);
    }
}

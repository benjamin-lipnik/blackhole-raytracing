import * as Util from "./util.js";

// This renderer just displays a texture to a canvas.
// It's main purpose is sampling the texture given from other renderers.
export default class DisplayRenderer {
    constructor(canvas) {
        this.canvas = canvas;
    }
    async initRenderer(config) {
        this.initCanvas(config);
	    this.pipeline = await this.initPipeline(config);
        this.sampler = config.device.createSampler({
            label: "DisplayRenderer sampler",
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "nearest",
        });
    }
    initCanvas(config) {
        this.context = this.canvas.getContext("webgpu");
        this.context.configure({
            device: config.device,
            format: config.format,
            alphaMode: "opaque"
        });
    }
    setResolution(config, source_texture) {
        if(!this.pipeline) {
            return;
        }

        this.bind_group = config.device.createBindGroup({
            label: "DisplayRenderer bindgroup",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: source_texture.createView() },
                { binding: 1, resource: this.sampler },
            ],
        });
    }
    async initPipeline(config) {
        let shader_code = await Util.loadFileString("./src/shaders/DisplayRenderer.wgsl");

        return await config.device.createRenderPipelineAsync({
    		label: "DisplayRenderer pipeline",
            layout: "auto",
            vertex: {
                entryPoint: "vs_main",
                module: config.device.createShaderModule({
    				label: "DisplayRenderer vs",
    				code: shader_code
    			}),
            },
            fragment: {
                entryPoint: "fs_main",
                module: config.device.createShaderModule({
    				label: "DisplayRenderer fs",
    				code: shader_code
    			}),
                targets: [ { format: config.format } ]
            },
            primitive: {
    			topology: "triangle-strip",
    			cullMode: "back",
                frontFace: "ccw"
    		}
        });
    }
    async render(config) {
        const commandEncoder = config.device.createCommandEncoder();
        const view = this.context.getCurrentTexture().createView();
        const renderPassDescriptor = {
            label: "DisplayRenderer render pass",
            colorAttachments: [{
                    view: view,
                    clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },
                    loadOp: "clear", // clear/load
                    storeOp: "store" // store/discard
                }]
        };
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
	    passEncoder.setBindGroup(0, this.bind_group);
        passEncoder.draw(3);
        passEncoder.end();
        config.device.queue.submit([commandEncoder.finish()]);
        await config.device.queue.onSubmittedWorkDone();
    }
}

import { mat4 } from "./lib/glMatrix.js";

export function resolutionRound(width) {
    width = Math.round(width);
    width += width % 2;
    let result = width >>> 0;
    return Math.min(result, 8192);
}

// WebGL to WebGPU matrix (webgl uses z ndc coordinate ranged in [-1,1], meanwhile
// webgpu uses ndc z axis ranged in [0,1]. glMatrix uses WebGL convetion so some
// matrices like mat4.perspective must be transformed into a smaller range.
export const web_gl_to_web_gpu = mat4.fromValues(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0.5, 0,
    0, 0, 0.5, 1
);
export function toWebGPU_NDC(proj) {
	// Apply OpenGL-to-WebGPU Z remapping
	mat4.multiply(proj, web_gl_to_web_gpu, proj);
}

// Loading files into strings
export async function loadFileString(filename) {
	return await (await fetch(filename)).text();
}

// Loading images into textures (https://webgpufundamentals.org/webgpu)
export async function createTextureFromImages(device, urls, options) {
    const images = await Promise.all(urls.map(loadImageBitmap));
    return createTextureFromSources(device, images, options);
}

export async function loadImageBitmap(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none' });
}

export function copySourcesToTexture(device, texture, sources, {flipY} = {}) {
    sources.forEach((source, layer) => {
        device.queue.copyExternalImageToTexture(
            { source, flipY, },
            { texture, origin: [0, 0, layer] },
            { width: source.width, height: source.height },
        );
    });
    if (texture.mipLevelCount > 1) {
        generateMips(device, texture);
    }
}

export function createTextureFromSources(device, sources, options = {}) {
    // Assume are sources all the same size so just use the first one for width and height
    const source = sources[0];
    const texture = device.createTexture({
        format: 'rgba8unorm',
        mipLevelCount: options.mips ? numMipLevels(source.width, source.height) : 1,
        size: [source.width, source.height, sources.length],
        usage: GPUTextureUsage.TEXTURE_BINDING |
                     GPUTextureUsage.COPY_DST |
                     GPUTextureUsage.RENDER_ATTACHMENT,
    });
    copySourcesToTexture(device, texture, sources, options);
    return texture;
}

export function numMipLevels(...sizes) {
    const maxSize = Math.max(...sizes);
    return 1 + Math.log2(maxSize) | 0;
}

export const generateMips = (() => {
    let sampler;
    let module;
    const pipelineByFormat = {};

    return function generateMips(device, texture) {
        if (!module) {
            module = device.createShaderModule({
                label: 'textured quad shaders for mip level generation',
                code: `
                    struct VSOutput {
                        @builtin(position) position: vec4f,
                        @location(0) texcoord: vec2f,
                    };

                    @vertex fn vs(
                        @builtin(vertex_index) vertexIndex : u32
                    ) -> VSOutput {
                        let pos = array(

                            vec2f( 0.0,    0.0),    // center
                            vec2f( 1.0,    0.0),    // right, center
                            vec2f( 0.0,    1.0),    // center, top

                            // 2st triangle
                            vec2f( 0.0,    1.0),    // center, top
                            vec2f( 1.0,    0.0),    // right, center
                            vec2f( 1.0,    1.0),    // right, top
                        );

                        var vsOutput: VSOutput;
                        let xy = pos[vertexIndex];
                        vsOutput.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
                        vsOutput.texcoord = vec2f(xy.x, 1.0 - xy.y);
                        return vsOutput;
                    }

                    @group(0) @binding(0) var ourSampler: sampler;
                    @group(0) @binding(1) var ourTexture: texture_2d<f32>;

                    @fragment fn fs(fsInput: VSOutput) -> @location(0) vec4f {
                        return textureSample(ourTexture, ourSampler, fsInput.texcoord);
                    }
                `,
            });

            sampler = device.createSampler({
                minFilter: 'linear',
                magFilter: 'linear',
            });
        }

        if (!pipelineByFormat[texture.format]) {
            pipelineByFormat[texture.format] = device.createRenderPipeline({
                label: 'mip level generator pipeline',
                layout: 'auto',
                vertex: {
                    module,
                },
                fragment: {
                    module,
                    targets: [{ format: texture.format }],
                },
            });
        }
        const pipeline = pipelineByFormat[texture.format];

        const encoder = device.createCommandEncoder({
            label: 'mip gen encoder',
        });

        for (let baseMipLevel = 1; baseMipLevel < texture.mipLevelCount; ++baseMipLevel) {
            for (let layer = 0; layer < texture.depthOrArrayLayers; ++layer) {
                const bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: sampler },
                        {
                            binding: 1,
                            resource: texture.createView({
                                dimension: '2d',
                                baseMipLevel: baseMipLevel - 1,
                                mipLevelCount: 1,
                                baseArrayLayer: layer,
                                arrayLayerCount: 1,
                            }),
                        },
                    ],
                });

                const renderPassDescriptor = {
                    label: 'our basic canvas renderPass',
                    colorAttachments: [
                        {
                            view: texture.createView({
                                dimension: '2d',
                                baseMipLevel: baseMipLevel,
                                mipLevelCount: 1,
                                baseArrayLayer: layer,
                                arrayLayerCount: 1,
                            }),
                            loadOp: 'clear',
                            storeOp: 'store',
                        },
                    ],
                };

                const pass = encoder.beginRenderPass(renderPassDescriptor);
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(6);    // call our vertex shader 6 times
                pass.end();
            }
        }
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
    };
})();

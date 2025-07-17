import * as Scene from "./scene.js";
import DisplayRenderer from "./DisplayRenderer.js";
import RaytracingRendererFS from "./RaytracingRendererFS.js";
import * as UserLogic from "./userLogic.js";
import * as Util from "./util.js";

async function initWebGPU(config) {
    if(!navigator.gpu) {
        console.log("WebGPU not supported.");
        return false;
    }
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
        antialias: false,
    });
    if (!adapter) {
        console.log("No Adapter Found");
        return false;
    }

    config.device = await adapter.requestDevice();
    config.format = navigator.gpu.getPreferredCanvasFormat();

    // I don't acutally care about css pixel ratio (or maybe just very little).
    // config.device_pixel_ratio = window.devicePixelRatio || 1;
    return true;
}
function getCanvas() {
    const canvas = document.querySelector("canvas");
    if (!canvas)
        throw new Error("No Canvas");
    return canvas;
}

function linearMap(x, inMin, inMax, outMin, outMax) {
    return ((x - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}
function percentMap(x, out_min, out_max) {
    return linearMap(x, 0, 100, out_min, out_max);
}

function scale_function_get_inverse_value(scale_function, value) {
    let i = 0;
    for(; i <= 100; ++i) {
        if(scale_function(i) >= value)
            break;
    }
    return i;
}

async function init() {
    let canvas = getCanvas();

    const initial_scaling_factor = 0.5;
    let config = {
        width:  Util.resolutionRound(canvas.clientWidth *  initial_scaling_factor),
        height: Util.resolutionRound(canvas.clientHeight * initial_scaling_factor),
        time: 0,
        last_render_time: 0,
        last_input_time: 0,
        last_resolution_change_time: 0,
        camera_fov: 1.2,
        fps_cap: 60,
        render_flat: false,
        renderer_index: 0,
        rerender: false,
        highres_drawn: true, // This prevents rendering in hight res before resize observer kicks in.
        resolution_change: false,
        uniforms: {
            g_min_step: 0.15,
            g_rs: 0.5,
            g_temperature: 0.5,
            g_disk_inner_multiplier: 3,
            g_disk_outer_multiplier: 8,
            g_falloff_rate: 10.0,
            g_beam_exponent: 0.5,
            g_redshift: 1.0,
            scaling_factor: initial_scaling_factor,
        },
    };

    let sliders = [
        { html_element: null, id: "temp_slider",       u_name: "g_temperature",           scale_function: (x) => percentMap(x,0,1) },
        { html_element: null, id: "beaming_slider",    u_name: "g_beam_exponent",         scale_function: (x) => percentMap(x, 0.1,5) },
        { html_element: null, id: "redshift_slider",   u_name: "g_redshift",              scale_function: (x) => percentMap(x, 0,2) },
        { html_element: null, id: "r_outer_slider",    u_name: "g_disk_outer_multiplier", scale_function: (x) =>
            Math.max(percentMap(x, 3, 12), config.uniforms.g_disk_inner_multiplier*1.1) },
        { html_element: null, id: "rs_slider",         u_name: "g_rs",                    scale_function: (x) => percentMap(x, 0.1, 3) },
        { html_element: null, id: "step_slider",       u_name: "g_min_step",              scale_function: (x) => percentMap(x, 0.02, 0.3) },
        { html_element: null, id: "resolution_slider", u_name: "scaling_factor",          scale_function:
            function(x) {
                config.resolution_change = true;
                config.last_resolution_change_time = config.time;
                return percentMap(x, 0.3, 6);
            }
        }
    ];

    function configureSliders() {
        for (let s of sliders) {
            if(!s.html_element) {
                s.html_element = document.getElementById(s.id);
            }
            s.html_element.value = scale_function_get_inverse_value(s.scale_function, config.uniforms[s.u_name]);
            s.html_element.oninput =
                function(a) {
                    let percent_val = a.target.value;
                    config.uniforms[s.u_name] = s.scale_function(percent_val);
                    config.rerender = true;
                    config.last_input_time = config.time;
                };
        }
        return sliders;
    }

    configureSliders();

    if(!await initWebGPU(config)) {
        alert("Ta brskalnik ne podpira WebGPU. Poskusi z drugim brskalnikom");
        return;
    }
    await Scene.init(config);

    let renderers = [
        new RaytracingRendererFS("./src/shaders/Relativistic.wgsl", true),
        new RaytracingRendererFS("./src/shaders/Relativistic.wgsl"),
        new RaytracingRendererFS("./src/shaders/Flat.wgsl", true),
        new RaytracingRendererFS("./src/shaders/Flat.wgsl")
    ];

    let output_textures = new Array(renderers.length).fill(null);
    for (let i = 0; i < renderers.length; ++i) {
        await renderers[i].initRenderer(config);
    }

    let display = new DisplayRenderer(canvas);
    await display.initRenderer(config);
    // console.log(display);
    console.log(config);

    async function renderersSetResolution() {
        canvas.width  = Util.resolutionRound(canvas.clientWidth );
        canvas.height = Util.resolutionRound(canvas.clientHeight);
        let aspect = canvas.width / canvas.height;

        if(aspect >= 1.0) {
            config.width  = Util.resolutionRound(canvas.clientWidth  * config.uniforms.scaling_factor);
            config.height = Util.resolutionRound(config.width / aspect);
        }else {
            config.height = Util.resolutionRound(canvas.clientHeight * config.uniforms.scaling_factor);
            config.width  = Util.resolutionRound(config.height * aspect);
        }

        for (let i = 0; i < renderers.length; ++i) {
            output_textures[i] = renderers[i].setResolution(config);
        }
        display.setResolution(config, output_textures[config.renderer_index]);
    }
    await renderersSetResolution();
    const observer = new ResizeObserver( async entries => {
        config.resolution_change = true;
        config.last_resolution_change_time = config.time;
    });
    observer.observe(canvas);
    UserLogic.init(swapRenderers);

    function setRenderer(index) {
        config.renderer_index = (index) % renderers.length;
        display.setResolution(config, output_textures[config.renderer_index]);
    }
    function swapRenderers() {
        config.render_flat = !config.render_flat;
        config.highres_drawn = false;
        config.renderer_index = 2*Number(config.render_flat);
        setRenderer(config.renderer_index);
        config.rerender = true;
    }
    // function sleep(ms) {
    //     return new Promise(resolve => setTimeout(resolve, ms));
    // }

    let prev_hires_drawn = false;
    async function frame(_time) {
        config.time = _time;
        requestAnimationFrame(frame);

        if(config.resolution_change) {
            config.rerender = false;
        }

        let reset_resolution = config.resolution_change && (config.time - config.last_resolution_change_time) > 650;
        if(reset_resolution) {
            config.resolution_change = false;
            config.rerender = true;
            config.last_resolution_change_time = config.time;
            await renderersSetResolution();
            return;
        }

        let skip_frame = !config.rerender;
        let input_active = UserLogic.update(config.time, (config.highres_drawn && !prev_hires_drawn));
        prev_hires_drawn = config.highres_drawn;
        if(input_active) {
            config.last_input_time = config.time;
        }

        let render_hires = !config.highres_drawn && (config.time - config.last_input_time) > 650;
        let over_fps_limit = ((config.time - config.last_render_time) < (1000.0 / config.fps_cap));
        let do_rendering = (config.rerender || input_active || render_hires) && !over_fps_limit;
        if(!do_rendering) {
            return;
        }

        config.rerender = false;
        config.highres_drawn = render_hires;
        config.renderer_index = 2*Number(config.render_flat) + Number(render_hires);

        let tmp_step_size = config.uniforms.g_min_step;
        if(!render_hires) {
            // Render with highest defined step size
            config.uniforms.g_min_step = sliders.find(slider => slider.id == "step_slider").scale_function(100);
        }
        setRenderer(config.renderer_index);
        await render();
        config.uniforms.g_min_step = tmp_step_size;
        config.last_render_time = config.time;
        // console.log("Render: ", render_hires?"High-res":"preview");
    }
    async function render() {
        await renderers[config.renderer_index].render(config);
        await display.render(config);
    }
    await frame(config.time);
}

init();

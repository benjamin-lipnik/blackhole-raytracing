import { mat4, vec3, quat } from "./lib/glMatrix.js";
import * as Util from "./util.js";

export class Node {
    constructor(pos = vec3.create(), rot = quat.create(), scale = vec3.fromValues(1,1,1), name="object") {
        this.position = pos;
        this.rotation = rot;
        this.scale = scale;
        this.name = name;
    }
    getNodeMatrix() {
    	let m = mat4.create();
        mat4.fromRotationTranslationScale(m, this.rotation, this.position, this.scale);
        return m;
    }
    getInverseNodeMatrix() {
    	let m = mat4.create();
        mat4.invert(m,this.getNodeMatrix());
        return m;
    }
}

export let camera = new Node(vec3.fromValues(-10,1,2), quat.fromEuler(quat.create(), 90,0,-90), vec3.fromValues(1,1,1), name="camera");
export let skybox_texture = null;
export let skybox_texture_view = null;
export let blackbody_color_texture = null;
export let blackbody_color_texture_view = null;

export async function init(config) {
    skybox_texture = await Util.createTextureFromImages(config.device,
        [
          "./assets/cubemap/sky2/px.png",
          "./assets/cubemap/sky2/nx.png",
          "./assets/cubemap/sky2/py.png",
          "./assets/cubemap/sky2/ny.png",
          "./assets/cubemap/sky2/pz.png",
          "./assets/cubemap/sky2/nz.png",
        ], {mips: true, flipY: false}
    );
    skybox_texture_view = skybox_texture.createView({dimension: "cube"});

    blackbody_color_texture = await Util.createTextureFromImages(config.device,
        [
          "./assets/blackbody.png",
        ], {mips: false, flipY: false}
    );
    blackbody_color_texture_view = blackbody_color_texture.createView();
}

export const cameraPerspective = (() => {
    let fov = -1;
    let ar = -1;
    let matrix = null;

    return function cameraPerspective(camera_fov, aspect_ratio) {
        if(camera_fov != fov || aspect_ratio != ar) {
            fov = camera_fov;
            ar = aspect_ratio;
            matrix = mat4.perspective(mat4.create(), fov, ar, 0.1, 500);
		    Util.toWebGPU_NDC(matrix);
            return [true, matrix];
        }
        return [false, matrix];
    }
})();

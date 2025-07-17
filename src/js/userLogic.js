import * as Scene from "./scene.js";
import { mat4, vec3, vec4, quat } from "./lib/glMatrix.js";

export let inputAxisRight = 0;
export let inputAxisFwd = 0;
export let inputAxisUp = 0;
let keys = new Array(6).fill(0);
let last_update = 0;
let _mouse_dx = 0;
let _mouse_dy = 0;
let mouse_sensitivity = 0.1;
let pointer_locked = false;

function getPointerLockElement() {
    return document.pointerLockElement || document.webkitPointerLockElement;
}

export function init(swap_rendereres_callback) {
    let key_right = ["d", "D", "ArrowRight"];
    let key_left  = ["a", "A", "ArrowLeft"];
    let key_up    = ["w", "W", "ArrowUp"];
    let key_down  = ["s", "S", "ArrowDown"];
    let key_q     = ["q", "Q"];
    let key_n     = ["n", "N"];
    let key_space = [" "];
    let key_shift = ["Shift"];

    document.onkeydown = function(evt) {
    	let key = evt.key;
    	// console.log(key);
    	keys[0] |= (key_right.includes(key));
    	keys[1] |= (key_left.includes(key));
    	keys[2] |= (key_up.includes(key));
    	keys[3] |= (key_down.includes(key));
    	keys[4] |= (key_space.includes(key));
    	keys[5] |= (key_shift.includes(key));

        if(key_q.includes(key)) {
            document.exitPointerLock();
        }
        if(key_n.includes(key)) {
            swap_rendereres_callback();
        }
        if(pointer_locked) {
            evt.preventDefault();
        }
    }
    document.onkeyup = function(evt) {
        let key = evt.key;
        keys[0] &= !(key_right.includes(key));
        keys[1] &= !(key_left.includes(key));
        keys[2] &= !(key_up.includes(key));
        keys[3] &= !(key_down.includes(key));
        keys[4] &= !(key_space.includes(key));
        keys[5] &= !(key_shift.includes(key));
    }

    const canvas = document.querySelector("canvas");

    document.addEventListener('pointerlockchange', () => {
        pointer_locked = getPointerLockElement() != null;
        // console.log("locked:", pointer_locked);
    });
    canvas.addEventListener("mousedown", (event) => {
        canvas.requestPointerLock();
    });
    canvas.addEventListener("mouseup", (event) => {
    });
    canvas.addEventListener("mousemove", (event) => {
        // console.log(event);
        _mouse_dx += event.movementX * mouse_sensitivity;
        _mouse_dy += event.movementY * mouse_sensitivity;
    });
}

export function update(time, ignore_input=false) {
    let mouse_dx = _mouse_dx;
    let mouse_dy = _mouse_dy;
    _mouse_dx = 0;
    _mouse_dy = 0;

	let delta_time = time - last_update;
	last_update = time;

    if(!pointer_locked) {
        return false;
    }

    inputAxisRight = (keys[0] - keys[1]);
    inputAxisFwd   = (keys[2] - keys[3]);
    inputAxisUp    = (keys[4] - keys[5]);

    let input_active = (mouse_dx || mouse_dy || inputAxisRight || inputAxisFwd || inputAxisUp);
    if (!input_active){
        return false;
    }
    if(ignore_input)
        return input_active;

    let inverse_view = mat4.clone(Scene.camera.getNodeMatrix());
	inverse_view[12] = 0;
	inverse_view[13] = 0;
	inverse_view[14] = 0;

    let move = vec3.fromValues(inputAxisRight, inputAxisUp, -inputAxisFwd);
    vec3.transformMat4(move, move, inverse_view);

    vec3.scale(move, move, 0.002 * delta_time);
    vec3.add(Scene.camera.position, Scene.camera.position, move);

    let rotate_y = -mouse_dx * 0.0005 * delta_time;
    let rotate_x = -mouse_dy * 0.0005 * delta_time;
	quat.rotateX(Scene.camera.rotation, Scene.camera.rotation, rotate_x);
	quat.rotateY(Scene.camera.rotation, Scene.camera.rotation, rotate_y);

    return true;
}

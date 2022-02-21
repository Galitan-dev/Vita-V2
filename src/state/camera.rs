// IMPORTS

use super::OPENGL_TO_WGPU_MATRIX;
use cgmath::{InnerSpace, SquareMatrix};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent},
};

// STRUCTURES

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// The informations about the camera that will be passed to the shader
pub struct CameraUniform {
    view_position: [f32; 4],
    pers_view_proj: [[f32; 4]; 4],
    ortho_view_proj: [[f32; 4]; 4],
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

// IMPLEMENTATIONS

impl Camera {
    pub fn build_view_projection_matrix(
        &self,
        size: PhysicalSize<u32>,
    ) -> [cgmath::Matrix4<f32>; 2] {
        // Camera Position & Rotation
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // Perspective
        let pers_proj =
            cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        // Orthographic
        let dist = self.eye - self.target;
        let scale = (dist.x.powi(2) + dist.y.powi(2) + dist.z.powi(2)).sqrt();
        let ortho_proj = cgmath::ortho(
            size.width as f32 * -scale / 100.0,
            size.width as f32 * scale / 100.0,
            size.height as f32 * -scale / 100.0,
            size.height as f32 * scale / 100.0,
            self.znear,
            self.zfar,
        );

        // Wgpu coordinate system
        [
            OPENGL_TO_WGPU_MATRIX * pers_proj * view,
            OPENGL_TO_WGPU_MATRIX * ortho_proj * view,
        ]
    }
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            pers_view_proj: cgmath::Matrix4::identity().into(),
            ortho_view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, size: PhysicalSize<u32>) {
        let view_proj = camera.build_view_projection_matrix(size);
        self.pers_view_proj = view_proj[0].into();
        self.ortho_view_proj = view_proj[1].into();

        self.view_position = camera.eye.to_homogeneous().into();
    }
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::Z | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Q | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the fowrard/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

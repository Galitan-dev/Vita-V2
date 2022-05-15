pub mod instance;
pub mod model;
pub mod texture;

use cgmath::Rotation3;
use noise::{NoiseFn, Perlin};
use std::path;
use wgpu::util::DeviceExt;

pub const NUM_INSTANCES_PER_ROW: u32 = 100;
pub const SPACE_BETWEEN: f32 = 1.9;
pub const WAVE_AMPLITUIDE: f32 = 2.0;

pub struct Object {
    pub instances: Vec<instance::Instance>,
    pub instance_buffer: wgpu::Buffer,
    pub model: model::Model,
}

impl Object {
    pub fn new(
        model_name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let res_dir = path::Path::new(env!("OUT_DIR")).join("res");

        let model = model::Model::load(
            device,
            queue,
            layout,
            res_dir.join("models").join(model_name).join("model.obj"),
        )
        .expect(&("Could not load `".to_owned() + model_name + "` model"));

        let perlin = Perlin::new();
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let position = cgmath::Vector3 {
                        x: SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0),
                        y: (perlin.get([x as f64 / 10.0, z as f64 / 10.0]) as f32
                            * WAVE_AMPLITUIDE)
                            .round()
                            * SPACE_BETWEEN,
                        z: SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0),
                    };

                    let rotation = cgmath::Quaternion::from_axis_angle(
                        cgmath::Vector3::unit_z(),
                        cgmath::Deg(0.0),
                    );

                    instance::Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        let instance_data = instances
            .iter()
            .map(instance::Instance::to_raw)
            .collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            model,
            instance_buffer,
            instances,
        }
    }
}

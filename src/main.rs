use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use winit::window::Window;

use noise::{NoiseFn, Perlin};

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        color: [0.5, 0.0, 0.5],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        color: [0.5, 0.0, 0.5],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        color: [0.5, 0.0, 0.5],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        color: [0.5, 0.0, 0.5],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        color: [0.5, 0.0, 0.5],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

const CAT_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.4, 0.8, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [-0.4, 0.6, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [-0.2, 0.6, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [0.4, 0.8, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [0.4, 0.6, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [0.2, 0.6, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [-0.4, 0.1, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.4, 0.1, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [-0.3, 0.1, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [-0.5, -0.6, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.0, -0.8, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.3, 0.1, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.5, -0.6, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.4, -0.4, 0.0],
        color: [0.9, 0.7, 0.0],
    },
    Vertex {
        position: [0.8, -0.3, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [0.7, -0.35, 0.0],
        color: [0.0, 0.1, 0.3],
    },
    Vertex {
        position: [0.65, -0.2, 0.0],
        color: [0.0, 0.1, 0.3],
    },
];

const CAT_INDICES: &[u16] = &[
    0, 1, 2, 3, 5, 4, 1, 6, 4, 4, 6, 7, 8, 9, 10, 8, 10, 11, 11, 10, 12, 13, 12, 14, 16, 15, 14,
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // This parts should be removed when we start creating the API
    background_color: wgpu::Color,
    perlin: Perlin,
    color_gaps: [f64; 3],
    chroma_render_pipeline: wgpu::RenderPipeline,
    key_state: KeyState,
    cat_vertex_buffer: wgpu::Buffer,
    cat_index_buffer: wgpu::Buffer,
    cat_num_indices: u32,
}

struct KeyState {
    space: bool,
    enter: bool,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let chroma_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Chroma Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("chroma_shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let chroma_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let chroma_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Chroma Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &chroma_shader,
                    entry_point: "vs_main",
                    buffers: &[chroma_buffer_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &chroma_shader,
                    entry_point: "fs_main",
                    targets: &[wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        let cat_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cat Vertex Buffer"),
            contents: bytemuck::cast_slice(CAT_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let cat_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cat Index Buffer"),
            contents: bytemuck::cast_slice(CAT_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let cat_num_indices = CAT_INDICES.len() as u32;

        let background_color = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        let perlin = Perlin::new();
        let color_gaps = rand::random();

        let key_state = KeyState {
            space: false,
            enter: false,
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,

            background_color,
            perlin,
            color_gaps,
            chroma_render_pipeline,
            key_state,
            cat_vertex_buffer,
            cat_index_buffer,
            cat_num_indices,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (
                    position.x / self.size.width as f64,
                    position.y / self.size.height as f64,
                );
                self.background_color = wgpu::Color {
                    r: self.perlin.get([x, y, self.color_gaps[0]]).abs(),
                    g: self.perlin.get([x, y, self.color_gaps[1]]).abs(),
                    b: self.perlin.get([x, y, self.color_gaps[2]]).abs(),
                    a: 1.,
                };

                true
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    },
                ..
            } => match virtual_keycode {
                Some(VirtualKeyCode::Space) => {
                    self.key_state.space = state == &ElementState::Pressed;
                    true
                }
                Some(VirtualKeyCode::Return) => {
                    self.key_state.enter = state == &ElementState::Pressed;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.background_color),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        let render_pipeline = if self.key_state.space {
            &self.chroma_render_pipeline
        } else {
            &self.render_pipeline
        };

        let (vertex_slice, index_slice, num_indices) = if self.key_state.enter {
            (
                self.cat_vertex_buffer.slice(..),
                self.cat_index_buffer.slice(..),
                self.cat_num_indices,
            )
        } else {
            (
                self.vertex_buffer.slice(..),
                self.index_buffer.slice(..),
                self.num_indices,
            )
        };

        render_pass.set_pipeline(render_pipeline);
        render_pass.set_vertex_buffer(0, vertex_slice);
        render_pass.set_index_buffer(index_slice, wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..num_indices, 0, 0..1);

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = pollster::block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &&mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}

use noise::{NoiseFn, Perlin};
use wgpu::util::DeviceExt;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod texture;

// Repetive parts are simplified with this macro
macro_rules! vertex {
    // Manually set texture coordinates
    ( $x:expr, $y:expr, $tex_x:expr, $tex_y:expr ) => {
        Vertex {
            position: [$x, $y, 0.0],
            tex_coords: [$tex_x, 1.0 - $tex_y],
        }
    };
    // Automatically calculate the texture coordinates from scale and position
    ( $x:expr, $y:expr, $off_x:expr, $off_y:expr, $scale: expr ) => {
        vertex!(
            $x,
            $y,
            ($x + 0.5 * $scale + $off_x) / $scale,
            ($y + 0.5 * $scale + $off_y) / $scale
        )
    };
}

// A simple shape of three triangles
const VERTICES: &[Vertex] = &[
    vertex!(-0.0868241, 0.49240386, 0.4131759, 0.99240386),
    vertex!(-0.49513406, 0.06958647, 0.0048659444, 0.56958647),
    vertex!(-0.21918549, -0.44939706, 0.28081453, 0.05060294),
    vertex!(0.35966998, -0.3473291, 0.85967, 0.1526709),
    vertex!(0.44147372, 0.2347359, 0.9414737, 0.7347359),
];
// Save duplicated data
const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

// Texture configuration
const OFFSET_X: f32 = 0.0;
const OFFSET_Y: f32 = -0.4;
const SCALE: f32 = 1.2;
// A complicated shape of about ten triangles
const CAT_VERTICES: &[Vertex] = &[
    vertex!(-0.4, 0.8, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(-0.4, 0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(-0.2, 0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.4, 0.8, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.4, 0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.2, 0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(-0.4, 0.1, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.4, 0.1, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(-0.3, 0.1, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(-0.5, -0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.0, -0.8, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.3, 0.1, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.5, -0.6, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.4, -0.4, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.8, -0.3, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.7, -0.35, OFFSET_X, OFFSET_Y, SCALE),
    vertex!(0.65, -0.2, OFFSET_X, OFFSET_Y, SCALE),
];
// Also save duplicated data
const CAT_INDICES: &[u16] = &[
    0, 1, 2, 3, 5, 4, 1, 6, 4, 4, 6, 7, 8, 9, 10, 8, 10, 11, 11, 10, 12, 13, 12, 14, 16, 15, 14,
];

// The object that is sent to the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

// The vector layout for wgpu
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
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// Wgpu does not appear to be normal
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

// To manipulate the viewpoint
struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // Camera Position & Rotation
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // Perspective
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        // Wgpu coordinate system
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

// The most important part of the code
struct State {
    surface: wgpu::Surface,                // Where to draw
    device: wgpu::Device,                  // Information about the device and its GPU/CPU
    queue: wgpu::Queue,                    // A sort of memory queue
    config: wgpu::SurfaceConfiguration,    // The configuration of the surface
    size: winit::dpi::PhysicalSize<u32>,   // Needed to draw in the good dimensions/resolutions
    render_pipeline: wgpu::RenderPipeline, // Actions to perfrom when rendering
    vertex_buffer: wgpu::Buffer,           // The vertices that we give to the shader
    index_buffer: wgpu::Buffer,            // Needed to remove duplicated data
    num_indices: u32,                      // Number of vertices to draw for our current shape
    diffuse_bind_group: wgpu::BindGroup, // The bind group passed to the render pipeline and the shader
    diffuse_texture: texture::Texture,   // Needed in the future
    camera: Camera,                      // The camera needed to control the view point

    // This parts should be removed when we start creating the API
    // Not going to comment this
    background_color: wgpu::Color,
    perlin: Perlin,
    color_gaps: [f64; 3],
    chroma_render_pipeline: wgpu::RenderPipeline,
    key_state: KeyState,
    cat_vertex_buffer: wgpu::Buffer,
    cat_index_buffer: wgpu::Buffer,
    cat_num_indices: u32,
    cat_diffuse_bind_group: wgpu::BindGroup,
    cat_diffuse_texture: texture::Texture,
}

// Needed to store the pressed keys
struct KeyState {
    space: bool,
    enter: bool,
    a: bool,
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

        let diffuse_bytes = include_bytes!("logo.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "logo.png").unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            // SamplerBindingType::Comparison is only for TextureSampleType::Depth
                            // SamplerBindingType::Filtering if the sample_type of the texture is:
                            //     TextureSampleType::Float { filterable: true }
                            // Otherwise you'll get an error.
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let cat_diffuse_bytes = include_bytes!("cat.png");
        let cat_diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, cat_diffuse_bytes, "cat.png").unwrap();

        let cat_diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cat_diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cat_diffuse_texture.sampler),
                },
            ],
            label: Some("cat_diffuse_bind_group"),
        });

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
                bind_group_layouts: &[&texture_bind_group_layout],
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
            a: false,
        };

        let camera = Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
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
            diffuse_bind_group,
            diffuse_texture,
            camera,

            background_color,
            perlin,
            color_gaps,
            chroma_render_pipeline,
            key_state,
            cat_vertex_buffer,
            cat_index_buffer,
            cat_num_indices,
            cat_diffuse_bind_group,
            cat_diffuse_texture,
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
                Some(VirtualKeyCode::A) => {
                    self.key_state.a = state == &ElementState::Pressed;
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

        let diffuse_bind_group = if self.key_state.a {
            &self.cat_diffuse_bind_group
        } else {
            &self.diffuse_bind_group
        };

        render_pass.set_pipeline(render_pipeline);
        render_pass.set_bind_group(0, diffuse_bind_group, &[]);
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

// IMPORTS

use vita::{listen, Vita};
use winit::{
    event::{ElementState, Event, KeyboardInput, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// MAIN FUNCTION

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut vita = pollster::block_on(Vita::new(&window, "default", "cube", None));

    listen!(
        event_loop,
        vita,
        window,
        event,
        (|| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode,
                                ..
                            },
                        ..
                    } if virtual_keycode.is_some() => println!("{:?}", virtual_keycode.unwrap()),
                    _ => (),
                }
            }
            _ => (),
        })
    );
}

extern crate piston_window;
extern crate rand;

mod lib;

use lib::Perceptron;
use piston_window::*;

fn main() {
    let mut perceptron = Perceptron::new(3, 1.);
    // The activation function. Here a simple sign function is used.
    let activate = |x: f64| if x >= 0. { 1. } else { 0. };
    // Use 1 as a bias.
    let bias = 1.;

    // The y = f(x) function.
    let f = |x: f64| x;

    // Keep the points in a vector, to redraw them.
    let mut points: Vec<([f32; 4], f64, f64)> = Vec::new();

    // Create window.
    let mut window: PistonWindow = WindowSettings::new("Perceptron", [400, 400])
        .exit_on_esc(true).build().unwrap();

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g| {
            // Clear window.
            clear([1.; 4], g);
            // Draw seperation line.
            line(
                [0., 0., 0., 1.],
                1.,
                [0., f(0.), 400., f(400.)],
                c.transform,
                g
            );

            if let Some(button) = e.press_args() {
                use piston_window::Button::{Keyboard, Mouse};
                use piston_window::Key::Space;

                match button {
                    Keyboard(Space) | Mouse(_) => {
                        points.clear();
                    },
                    _ => (),
                }
            }

            // Draw perceptron's seperation line.
            let weights = perceptron.weights.clone();
            line(
                [0., 0., 0., 0.25],
                5.,
                [0., (-weights[2] - weights[0] * 0.) / weights[1],
                 400., (-weights[2] - weights[0] * 400.) / weights[1]],
                c.transform,
                g
            );

            // Create a random point.
            let x = 400. * rand::random::<f64>();
            let y = 400. * rand::random::<f64>();

            // Generate output.
            let output = perceptron.feed_forward(vec![x, y, bias], &activate);
            // Red if above the line, green if under the line.
            let color = if output == 1. {
                [0., 1., 0., 1.]
            } else {
                [1., 0., 0., 1.]
            };

            // Add point.
            points.push((color, x, y));

            // Draw all points.
            for point in points.iter() {
                ellipse(
                    point.0,
                    [point.1, point.2, 10., 10.],
                    c.transform,
                    g
                );
            }

            // Train the perceptron.
            let desired = if y > f(x) { 1. } else { 0. };
            perceptron.train((vec![x, y, bias], desired), &activate);
        });
    }
}

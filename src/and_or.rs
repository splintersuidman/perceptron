mod lib;

use lib::Perceptron;

fn main() {
    let bias = 1.0;
    let activate = |x: f64| if x > 0. { 1. } else { 0. };

    let or = or_function(bias);
    let and = and_function(bias);

    println!("0 && 0 = {}", and.feed_forward(vec![0.0, 0.0, bias], &activate));
    println!("1 && 0 = {}", and.feed_forward(vec![1.0, 0.0, bias], &activate));
    println!("0 && 1 = {}", and.feed_forward(vec![0.0, 1.0, bias], &activate));
    println!("1 && 1 = {}", and.feed_forward(vec![1.0, 1.0, bias], &activate));
    println!("0 || 0 = {}", or.feed_forward(vec![0.0, 0.0, bias], &activate));
    println!("1 || 0 = {}", or.feed_forward(vec![1.0, 0.0, bias], &activate));
    println!("0 || 1 = {}", or.feed_forward(vec![0.0, 1.0, bias], &activate));
    println!("1 || 1 = {}", or.feed_forward(vec![1.0, 1.0, bias], &activate));
}

// Train an or function.
fn or_function(bias: f64) -> Perceptron {
    let mut or = Perceptron::new(3, 1.);

    let activate = |x: f64| if x > 0. { 1. } else { 0. };

    for _ in 0..100 {
        or.train_multiple(
            vec![
                (vec![0.0, 0.0, bias], 0.0),
                (vec![1.0, 0.0, bias], 1.0),
                (vec![0.0, 1.0, bias], 1.0),
                (vec![1.0, 1.0, bias], 1.0),
            ],
            &activate,
        );
    }

    or
}

// Train an and function.
fn and_function(bias: f64) -> Perceptron {
    let mut and = Perceptron::new(3, 1.);

    let activate = |x: f64| if x > 0. { 1. } else { 0. };

    for _ in 0..100 {
        and.train_multiple(
            vec![
                (vec![0.0, 0.0, bias], 0.0),
                (vec![1.0, 0.0, bias], 0.0),
                (vec![0.0, 1.0, bias], 0.0),
                (vec![1.0, 1.0, bias], 1.0),
            ],
            &activate,
        );
    }

    and
}

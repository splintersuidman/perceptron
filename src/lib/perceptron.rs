extern crate rand;

#[derive(Debug)]
/// The `Perceptron` struct that
pub struct Perceptron {
    /// The amount of weights, and thus the amount of inputs.
    pub length: usize,
    /// The weights vector.
    pub weights: Vec<f64>,
    /// The learning rate changes what impact the training has.
    pub learning_rate: f64,
}

impl Perceptron {
    /// Create a new `Perceptron` with the length, and the learning rate.
    pub fn new(length: usize, learning_rate: f64) -> Self {
        let mut weights = Vec::new();
        for _ in 0..length {
            weights.push(rand::random());
        }

        Perceptron { length, weights, learning_rate }
    }

    /// Generate output from an input.
    pub fn feed_forward<F>(&self, input: Vec<f64>, activate: &F) -> f64
        where F: Fn(f64) -> f64
    {
        if input.len() != self.weights.len() {
            panic!("The lengths of the weights and the inputs do not match.");
        }
        let mut sum = 0f64;

        for i in 0..self.length {
            sum += input[i] * self.weights[i];
        }

        activate(sum)
    }

    /// Train the perceptron by providing the input and the desired output.
    pub fn train<F>(&mut self, input_desired: (Vec<f64>, f64), activate: &F) -> &Self
        where F: Fn(f64) -> f64
    {
        let (input, desired) = input_desired;
        let guess = self.feed_forward(input.clone(), activate);
        let error = desired - guess;

        for i in 0..self.length {
            self.weights[i] += self.learning_rate * error * input[i];
        }

        self
    }

    #[allow(dead_code)]
    /// Train the perceptron with a vector of inputs and desired outputs.
    pub fn train_multiple<F>(&mut self, inputs_desired: Vec<(Vec<f64>, f64)>, activate: &F) -> &Self
        where F: Fn(f64) -> f64
    {
        for input_desired in inputs_desired {
            self.train(input_desired, activate);
        }

        self
    }
}

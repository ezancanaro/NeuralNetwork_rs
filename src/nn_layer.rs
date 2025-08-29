use std::cmp::max;

/**
 * TO-DO: Rewrite Neurons as a Matrix instead of Vec
 */
//http://neuralnetworksanddeeplearning.com/chap2.html
//https://www.3blue1brown.com/lessons/backpropagation-calculus#title
use crate::nn_matrix::Matrix;

struct Neuron {
    value: f64,
}

impl Neuron {
    //activate(bias + sum(x_i * w_i))
    //Aplicada ao resultado da soma dos pesos * valor da camada anterior
    //somado ao viés da camada
}

pub trait ActivationFunction {
    fn activate(&self, val: f64) -> f64;
    fn derivative(&self, val: f64) -> f64;
}

pub struct Sigmoid {}
impl ActivationFunction for Sigmoid {
    fn activate(&self, val: f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-val))
    }
    fn derivative(&self, val: f64) -> f64 {
        let sigma = self.activate(val);
        sigma * (1.0 - sigma)
    }
}
pub struct Relu {}
impl ActivationFunction for Relu {
    fn activate(&self, val: f64) -> f64 {
        f64::max(0.0, val)
    }
    fn derivative(&self, val: f64) -> f64 {
        match val {
            0.0 | _ if val < 0.0 => 0.0,
            _ => 1.0,
        }
    }
}
// type Link = Box<Layer>;
pub struct Layer {
    neurons: Matrix,
    weighted_input: Matrix,
    deltas: Matrix,
    weights: Matrix,
    biases: Matrix,
    activation: Box<dyn ActivationFunction>, //Verificar como armazenar o objeto de trait
}

impl Layer {
    //Cria uma nova camada
    pub fn new(
        prev_layer_neurons: usize,
        layer_neurons: usize,
        activation_function: impl ActivationFunction + 'static, //Verificar se é a melhor forma de armazenar isso aqui
    ) -> Layer {
        Layer {
            neurons: Matrix::new(layer_neurons, 1),
            weighted_input: Matrix::new(layer_neurons, 1),
            deltas: Matrix::new(layer_neurons, 1),
            weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
            biases: Matrix::new_random(layer_neurons, 1),
            activation: Box::new(activation_function),
        }
    }

    pub fn deltas(&self) -> &Matrix {
        &self.deltas
    }

    pub fn propagate(&mut self, input_neurons: Matrix) {
        //activation = act_fn( bias + sum_i(input_neurons_i * weights_i) )
        // let weight_transpose = self.weights.transpose();
        let dot_product = &(self.weights) * &input_neurons; //A ordem importa (input * weights) geraria erro!
        let biased_values = dot_product + &self.biases;
        assert!(biased_values.rows() == self.neurons.rows());
        //Biased_values deve ser uma matriz nx1
        for i in 0..biased_values.rows() {
            //Armazena o resultado para a fase de backprop
            self.weighted_input[i][0] = biased_values[i][0]; //z
            self.neurons[i][0] = self.activation.activate(biased_values[i][0]);
        }
    }

    pub fn cost(&mut self, expected: Matrix) -> f64 {
        assert!(self.neurons.rows() == expected.rows());
        let mut sum = 0.0;
        for i in 0..self.neurons.rows() {
            sum += (self.neurons[i][0] - expected[i][0]).powi(2);
        }
        return 0.5 * sum;
    }
    pub fn cost_derivative(activation_val: f64, expected_val: f64) -> f64 {
        activation_val - expected_val
    }

    /**
     * http://neuralnetworksanddeeplearning.com/chap2.html
     * Final Layer (l): generate vector δ_l = hadamard_product(∇aC, derivative(z_l))
     * Hidden Layer: vector δ_l = hadamard_product((w_l+1)^T * δ_l+1, derivative(z_l))
     *
     * */
    pub fn backpropagate_output(&mut self, expected: Matrix) {
        for i in 0..self.weighted_input.rows() {
            let derivative = self.activation.derivative(self.weighted_input[i][0]);
            let cost_derivative = Layer::cost_derivative(self.neurons[i][0], expected[i][0]);
            self.deltas[i][0] = derivative * cost_derivative;
        }
    }

    pub fn weights_transpose(&self) -> Matrix {
        self.weights.transpose()
    }

    pub fn backpropagate_hidden(
        &mut self,
        next_layer_transpose: &Matrix,
        next_layer_deltas: &Matrix,
    ) {
        //TO-DO: tratar corretamente o tipo Option.
        let next_layer_transpose = next_layer_transpose;
        let dot_product = next_layer_transpose * next_layer_deltas;

        for i in 0..self.weighted_input.rows() {
            let derivative = self.activation.derivative(self.weighted_input[i][0]);
            self.deltas[i][0] = derivative * dot_product[i][0];
        }
    }
}

/**
 *  Copyright 2025 Eric Zancanaro
 *    
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
pub struct Identity {}
impl ActivationFunction for Identity {
    fn activate(&self, val: f64) -> f64 {
        val
    }
    fn derivative(&self, _: f64) -> f64 {
        1.0
    }
}
// type Link = Box<Layer>;
pub struct Layer<T: ActivationFunction> {
    neurons: Matrix,
    weighted_input: Matrix,
    deltas: Matrix,
    weights: Matrix,
    biases: Matrix,
    activation: T, //Verificar como armazenar o objeto de trait
}

impl<T: ActivationFunction> Layer<T> {
    //Cria uma nova camada
    pub fn new(
        prev_layer_neurons: usize,
        layer_neurons: usize,
        activation_function: T, //Verificar se é a melhor forma de armazenar isso aqui
    ) -> Layer<T> {
        Layer {
            neurons: Matrix::new(layer_neurons, 1),
            weighted_input: Matrix::new(layer_neurons, 1),
            deltas: Matrix::new(layer_neurons, 1),
            weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
            biases: Matrix::new(layer_neurons, 1),
            activation: activation_function,
        }
    }

    pub fn deltas(&self) -> &Matrix {
        &self.deltas
    }

    pub fn propagate(&mut self, input_neurons: &Matrix) {
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
            let cost_derivative = Layer::<T>::cost_derivative(self.neurons[i][0], expected[i][0]);
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

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope. (Rust Book)
    use super::*;
    #[test]
    fn test_propagate() {
        let input_n = 3;
        let layer1_n = 5;
        let layer2_n = 7;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut layer1 = Layer::new(input_n, layer1_n, Identity {});
        let mut layer2 = Layer::new(layer1_n, layer2_n, Identity {});

        let input_mock = Matrix::from_vec(input_n, 1, vec![1.0, 1.0, 1.0]);
        layer1.propagate(&input_mock);
        layer2.propagate(&layer1.neurons);

        let linear_transform =  &layer2.weights * &layer1.weights;
        let bias_transform = (&layer2.weights * &layer1.biases) + &layer2.biases;

        print!("Neurons:{}", layer2.neurons);
        let expected = (linear_transform * input_mock) + &bias_transform;
        print!("Expected: {}", expected);

        assert!(expected == layer2.neurons);
    }
}

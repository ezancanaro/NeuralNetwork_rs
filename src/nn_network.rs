use crate::nn_layer::ActivationFunction;
use crate::nn_layer::Gradient;
use crate::nn_layer::Layer;
use crate::nn_layer::Relu;
use crate::nn_matrix::Matrix;
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
use std::collections::VecDeque;
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(num_layers: usize, _learning_rate: f64) -> NeuralNetwork {
        NeuralNetwork {
            layers: Vec::with_capacity(num_layers),
            learning_rate: _learning_rate,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn cost_derivative_mse(x: f64, y: f64) -> f64 {
        2.0 * (x - y)
    }

    /* Original, com problema por conta do borrow checker.
    pub fn train(&mut self, input: Matrix) {
    assert!(!self.layers.is_empty());
    let input_layer = self.layers.get_mut(0).unwrap();
    assert!(input.rows() == input_layer.neuron_qty());
    input_layer.propagate(&input);
    let mut prev_layer = input_layer;
    for i in 1..self.layers.len() {
    let current_layer = self.layers.get_mut(i).unwrap();
    current_layer.propagate(prev_layer.neurons());
    }
    }
    */
    /**
     * Treinamento da rede neural. Recebe os dados de entrada,
     * propaga em toda a rede e executa o algoritmo de retropropagação.
     *
     */
    pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
        assert!(!self.layers.is_empty());
        let last_layer_index = self.layers.len() - 1;
        //Propaga a primeira camada
        self.layers[0].load_input(input);
        //Propaga as camadas remanescentes
        for i in 1..self.layers.len() {
            //Separa em 2 slices: [0..i) e [i..len)
            //Necessário para lidar com o borrow checker de Rust
            let (prev_layers, layers_to_propagate) = self.layers.split_at_mut(i);
            layers_to_propagate[0].propagate(prev_layers[i - 1].neurons());
        }
        //Limita o escopo dos slices para evitar erro de borrow na retropropagação
        let mut gradients: VecDeque<Gradient> = VecDeque::with_capacity(self.layers.len());
        {
            let (hidden_layers, output_layers) = self.layers.split_at_mut(last_layer_index);
            let gradient = output_layers[0].backpropagate_output_layer(
                &expected_output,
                hidden_layers[last_layer_index - 1].neurons(),
                &NeuralNetwork::cost_derivative_mse,
            );
            gradients.push_front(gradient);
        };

        for i in (1..last_layer_index).rev() {
            //slices [0..i) e [i..len()-1) (Novamente lidando com borrow checker)
            let (initial_layers, current_and_done_layers) = self.layers.split_at_mut(i);
            let (current_layer, done_layers) = current_and_done_layers.split_at_mut(1);

            let gradient = current_layer[0].backpropagate_hidden_layer(
                &done_layers[0].weights(),
                &gradients[0].delta,
                initial_layers[i - 1].neurons(),
            );
            gradients.push_front(gradient);
        }

        assert!(gradients.len() == self.layers.len() - 1);
        let adjustable_layers = self.layers.iter_mut().skip(1);
        //zip: agrupa 2 iteradores. O laço é finalizado quanto um deles chega ao fim.
        //No nosso caso, ambos terão o mesmo tamanho, dado o assert! acima.
        for (layer, gradient) in adjustable_layers.zip(gradients) {
            layer.adjust_parameters(gradient, self.learning_rate);
        }
    }
}

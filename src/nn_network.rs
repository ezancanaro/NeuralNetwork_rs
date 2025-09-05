use std::thread::current;

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
use crate::nn_layer::Layer;
use crate::nn_layer::Relu;
use crate::nn_matrix::Matrix;
struct NeuralNetwork {
    layers: Vec<Layer<Relu>>,
    gradient: Vec<f64>,
}

impl NeuralNetwork {
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
    pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
        assert!(!self.layers.is_empty());
        let last_layer_index = self.layers.len() - 1;
        //Propaga a primeira camada
        self.layers[0].propagate(&input);
        //Propaga as camadas remanescentes
        for i in 1..self.layers.len() {
            //Separa em 2 slices: [0..i) e [i..len)
            //Necessário para lidar com o borrow checker de Rust
            let (prev_layers, layers_to_propagate) = self.layers.split_at_mut(i);
            layers_to_propagate[0].propagate(prev_layers[i - 1].neurons());
        }
        //Limita o escopo dos slices para evitar erro de borrow na retropropagação
        {
            let (hidden_layers, output_layers) = self.layers.split_at_mut(last_layer_index);
            output_layers[0].backpropagate_output_layer(
                &expected_output,
                hidden_layers[last_layer_index - 1].neurons(),
                NeuralNetwork::cost_derivative_mse,
            );
        }
        for i in (2..last_layer_index).rev() {
            //slices [0..i) e [i..len()] (Novamente lidando com borrow checker)
            let (propagation_layers, done_layers) = self.layers.split_at_mut(i);
            //slices [0..i-1)] e [i-1)
            let (coming_layers, current_layers) = propagation_layers.split_at_mut(i - 1);
            current_layers[0]
                .backpropagate_hidden_layer(&done_layers[0], coming_layers[i - 2].neurons());
        }
        let (remaining_layers, done_layers) = self.layers.split_at_mut(2);


    }

    pub fn backpropagate(&mut self, expected: Matrix) {
        //TO-DO: tratar option corretamente
        let mut iterator = self.layers.iter_mut().rev();
        let mut last_layer = iterator.next().unwrap();
        // last_layer.backpropagate_output(expected);
        // for current_layer in iterator {
        //     let transpose = last_layer.weights_transpose();
        //     let deltas = last_layer.deltas();
        //     current_layer.backpropagate_hidden(&transpose, deltas);

        //     last_layer = current_layer;
        // }
    }
}

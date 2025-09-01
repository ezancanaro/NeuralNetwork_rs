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
use crate::nn_layer;
use crate::nn_matrix::Matrix;
struct NeuralNetwork {
    layers: Vec<nn_layer::Layer<nn_layer::Relu>>,
    gradient: Vec<f64>,
}

impl NeuralNetwork {
    pub fn backpropagate(&mut self, expected: Matrix) {
        //TO-DO: tratar option corretamente
        let mut iterator = self.layers.iter_mut().rev();
        let mut last_layer =iterator.next().unwrap();
        last_layer.backpropagate_output(expected);
        for current_layer in iterator {
            let transpose = last_layer.weights_transpose();
            let deltas = last_layer.deltas();
            current_layer.backpropagate_hidden(&transpose, deltas);
            
            last_layer = current_layer;
        }
    }
}

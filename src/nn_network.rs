use std::iter;

use crate::nn_layer;
use crate::nn_matrix;
struct NeuralNetwork {
    layers: Vec<nn_layer::Layer>,
    gradient: Vec<f64>,
}

impl NeuralNetwork {
    pub fn backpropagate(&mut self, expected: Vec<f64>) {
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

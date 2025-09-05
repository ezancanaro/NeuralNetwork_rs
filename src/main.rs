mod nn_layer;
mod nn_matrix;
mod nn_network;
use std::time::{Duration, Instant};

use nn_layer::Layer;
use nn_layer::Relu;

fn main() {
    println!("Inicializando Rede!");

    let activation_function = Relu {};
    let mut network = nn_network::NeuralNetwork::new(2, 0.8);

    println!("Adicionando Camadas!");
    let input_layer = Layer::new::<Relu>(784, 784);
    let hidden_layer1 = Layer::new::<Relu>(784, 128);
    let hidden_layer2 = Layer::new::<Relu>(128, 128);
    let output_layer = Layer::new::<Relu>(128, 10);

    network.add_layer(input_layer);
    network.add_layer(hidden_layer1);
    network.add_layer(hidden_layer2);
    network.add_layer(output_layer);

    let input = nn_matrix::Matrix::new_random(784, 1);
    let output = nn_matrix::Matrix::new_random(784, 1);
    println!("Iniciando treinamento...");
    
    let start = Instant::now();
    
    network.train(input, output);
    let duration = start.elapsed();

    println!("Training Time is: {:?}", duration);

}

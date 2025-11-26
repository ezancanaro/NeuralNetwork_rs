mod nn_emnist;
mod nn_layer;
mod nn_matrix;
mod nn_network;
mod nn_prev;
mod nn_prevL;
use std::net;
use std::time::{Duration, Instant};

use nn_layer::Layer;
use nn_layer::Relu;
use std::env;

use crate::nn_layer::{Sigmoid, Softmax};

fn label_to_vec(label: u8) -> Vec<f64> {
    let mut v: [f64; 10] = [0.0; 10];
    v[label as usize] = 1.0;
    v.to_vec()
}

fn vec_to_label(vec: &Vec<f64>) -> u8 {
    vec.iter()
        .enumerate()
        .reduce(|(acc_i, acc_v), (i, v)| if v > acc_v { (i, v) } else { (acc_i, acc_v) })
        .unwrap()
        .0 as u8
}

fn train_emnist(network: &mut nn_network::NeuralNetwork, max_samples: u32) {
    let mut parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-train-labels-idx1-ubyte",
        "emnist/emnist-digits-train-images-idx3-ubyte",
    );

    println!("Iniciando treinamento...");
    let start = Instant::now();
    let mut samples = 0;
    while parser.has_more() && samples < max_samples {
        let (img, label) = parser.read_next();
        //Normaliza o valor dos pixels para 0..1 dividindo por 255
        let vec64 = img.iter().map(|f| (*f as f64) / 255.0).collect();

        let input = nn_matrix::Matrix::from_vec(784, 1, vec64);
        let expected = nn_matrix::Matrix::from_vec(10, 1, label_to_vec(label));

        let output = network.classify(&input);
        let out_label = vec_to_label(output.data());
        // println!("Label: {}. Found: {}" label, out_label, output);

        network.train(input, expected);
        samples += 1;
        // print!(".");
        // if samples % 10 == 0 {
        //     let duration = start.elapsed();
        //     println!(
        //         "Training Time is: {:?}. Trained on {} samples",
        //         duration, samples
        //     );
        // }
    }
    let duration = start.elapsed();
    println!("Total Training Time is: {:?}", duration);
}

fn test_emnist_on_training(network: &mut nn_network::NeuralNetwork, max_samples: u32) {
    let mut test_parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-train-labels-idx1-ubyte",
        "emnist/emnist-digits-train-images-idx3-ubyte",
    );
    let mut right_classification = 0;
    let mut test_samples = 0;
    while test_parser.has_more() && test_samples < max_samples {
        let (img, label) = test_parser.read_next();
        //Normaliza o valor dos pixels para 0..1 dividindo por 255
        let vec64 = img.iter().map(|f| (*f as f64) / 255.0).collect();
        let input = nn_matrix::Matrix::from_vec(784, 1, vec64);

        let output = network.classify(&input);
        let out_label = vec_to_label(output.data());
        if out_label == label {
            right_classification += 1;
        }
        //println!("Label: {}. Found: {} Output:{}", label, out_label, output);
        test_samples += 1;
    }
    let accuracy = right_classification as f64 / test_samples as f64 * 100.0;

    println!(
        "Total Samples: {}. Right Classifications:{}. Accuracy: {}%",
        test_samples, right_classification, accuracy
    );
}

fn test_emnist(network: &mut nn_network::NeuralNetwork, max_samples: u32) {
    let mut test_parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-test-labels-idx1-ubyte",
        "emnist/emnist-digits-test-images-idx3-ubyte",
    );
    let mut right_classification = 0;
    let mut test_samples = 0;
    while test_parser.has_more() && test_samples < max_samples {
        let (img, label) = test_parser.read_next();
        //Normaliza o valor dos pixels para 0..1 dividindo por 255
        let vec64 = img.iter().map(|f| (*f as f64) / 255.0).collect();
        let input = nn_matrix::Matrix::from_vec(784, 1, vec64);

        let output = network.classify(&input);
        let out_label = vec_to_label(output.data());
        if out_label == label {
            right_classification += 1;
        }
        //println!("Label: {}. Found: {} Output:{}", label, out_label, output);
        test_samples += 1;
    }
    let accuracy = right_classification as f64 / test_samples as f64 * 100.0;

    println!(
        "Total Samples: {}. Right Classifications:{}. Accuracy: {}%",
        test_samples, right_classification, accuracy
    );
}

fn main() {
    println!("Inicializando Rede!");
    let args: Vec<String> = env::args().collect();
    let max_epochs = 10;
    let mut epoch = 0 ;
    let (training_samples, test_samples) = if args.len() > 2 {
        (
            args[1]
                .parse()
                .expect("Numero de Samples de treino deve ser INT"),
            args[2]
                .parse()
                .expect("Numero de Samples de teste deve ser INT"),
        )
    } else {
        (60000, 60000)
    };

    let mut network = nn_network::NeuralNetwork::new(2, 0.1);

    println!("Adicionando Camadas!");
    //let input_layer = Layer::new::<Relu>(784, 784);
    let hidden_layer1 = Layer::new::<Relu>(784, 128);
    let hidden_layer2 = Layer::new::<Relu>(128, 128);
    let hidden_layer3 = Layer::new::<Relu>(128, 128);
    let output_layer = Layer::new::<Sigmoid>(128, 10);

    //network.add_layer(input_layer);
    network.add_layer(hidden_layer1);
    network.add_layer(hidden_layer2);
    network.add_layer(hidden_layer3);
    network.add_layer(output_layer);

    while epoch < max_epochs {
        println!("Epoch {}.",epoch);
        println!("Training on EMNIST DataSet...");
        train_emnist(&mut network, training_samples);
        println!("Test on Training samples...");
        test_emnist_on_training(&mut network, test_samples);
        println!("Testing neural net...");
        test_emnist(&mut network, test_samples);
        epoch += 1;
    }
    


}

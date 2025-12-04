mod nn_emnist;
mod nn_layer;
mod nn_matrix;
mod nn_network;
use std::collections::VecDeque;
use std::time::Instant;

use nn_layer::Layer;
use std::env;

use crate::nn_layer::{Gradient, Sigmoid, Softmax};
use crate::nn_matrix::Matrix;

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

fn normalize_cast_f64(original: Vec<u8>) -> Vec<f64>{
    original.iter().map(|f| (*f as f64) / 255.0).collect()
} 

fn train_emnist(network: &mut nn_network::NeuralNetwork, max_samples: u32, mixing_f: fn(Vec<u8>)->Vec<f64>) {
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
        let vec64 = mixing_f(img);

        let input = nn_matrix::Matrix::from_vec(784, 1, vec64);
        let expected = nn_matrix::Matrix::from_vec(10, 1, label_to_vec(label));

        network.train(input, expected);
        samples += 1;
        // print!(".");
        if samples % 10000 == 0 {
            let duration = start.elapsed();
            println!(
                "\tTrained on {} samples. Training Time is: {:?}.",
                samples, duration
            );
        }
    }
    let duration = start.elapsed();
    println!("Total Training Time is: {:?}", duration);
}

fn batch_train_emnist(network: &mut nn_network::NeuralNetwork, max_samples: u32) {
    let mut parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-train-labels-idx1-ubyte",
        "emnist/emnist-digits-train-images-idx3-ubyte",
    );

    println!("Iniciando treinamento...");
    let start = Instant::now();
    let mut samples = 0;
    let mut gradients: VecDeque<Gradient> = VecDeque::with_capacity(network.num_layers());

    while parser.has_more() && samples < max_samples {
        let (img, label) = parser.read_next();
        //Normaliza o valor dos pixels para 0..1 dividindo por 255
        let vec64 = img.iter().map(|f| (*f as f64 / 255.0)).collect();

        let input = nn_matrix::Matrix::from_vec(784, 1, vec64);
        let expected = nn_matrix::Matrix::from_vec(10, 1, label_to_vec(label));

        network.train_batch(input, expected, &mut gradients);
        samples += 1;

        if samples % 1000 == 0 {
            println!("Averaging gradients...");
            gradients.iter_mut().for_each(|gradient| {
                println!(
                    "Weight Zero?{}. Bias Zero?{}.",
                    gradient.weight.is_zero(),
                    gradient.delta.is_zero()
                );
                println!("Gradient:{}", gradient.weight);
                gradient.weight.mut_map(|v| v / 1000.0);
                gradient.delta.mut_map(|v| v / 1000.0);
            });

            network.adjust_parameters(&mut gradients);

            gradients.iter_mut().for_each(|gradient| gradient.zero());
        }

        // print!(".");
        if samples % 10000 == 0 {
            let duration = start.elapsed();
            println!(
                "\tTrained on {} samples. Training Time is: {:?}.",
                samples, duration
            );
        }
    }
    let duration = start.elapsed();
    println!("Total Training Time is: {:?}", duration);
}

fn test_emnist_on_training(network: &mut nn_network::NeuralNetwork, max_samples: u32, mixing_f: fn(Vec<u8>)->Vec<f64>) {
    let mut test_parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-train-labels-idx1-ubyte",
        "emnist/emnist-digits-train-images-idx3-ubyte",
    );
    let mut right_classification = 0;
    let mut test_samples = 0;
    while test_parser.has_more() && test_samples < max_samples {
        let (img, label) = test_parser.read_next();
        //Normaliza o valor dos pixels para 0..1 dividindo por 255
        let vec64 = mixing_f(img); 
        
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

fn test_emnist(network: &mut nn_network::NeuralNetwork, max_samples: u32, mixing_f: fn(Vec<u8>)->Vec<f64>) {
    let mut test_parser = nn_emnist::Parser::setup(
        "emnist/emnist-digits-test-labels-idx1-ubyte",
        "emnist/emnist-digits-test-images-idx3-ubyte",
    );
    let mut right_classification = 0;
    let mut test_samples = 0;
    while test_parser.has_more() && test_samples < max_samples {
        let (img, label) = test_parser.read_next();
        let vec64 = mixing_f(img);
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
    let _max_epochs = 10;
    let mut epoch = 0;
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
        (280000, 280000)
    };

    let temp = vec![0.0];

    let mut cacheable_softmax: Softmax = Softmax {
        cached_zed: temp,
        cached_max: 0.0,
        cached_sum: 0.0,
    };

    let _closure = move |val: f64, z: &Vec<f64>| cacheable_softmax.activate(val, z);

    let mut network = nn_network::NeuralNetwork::new(4, 0.4);

    println!("Adicionando Camadas!");
    //let input_layer = Layer::new::<Relu>(784, 784);
    let hidden_layer1 = Layer::new::<Sigmoid>(784, 128);
    let hidden_layer2 = Layer::new::<Sigmoid>(128, 128);
    let hidden_layer3 = Layer::new::<Sigmoid>(128, 128);
    let output_layer = Layer::new::<Sigmoid>(128, 10);
    //Não implementei a derivada da Softmax por preguiça. Como é uma função matricial, precisaria estudar jacobianas para lembrar como isso é feito.
    //Para que a função softmax funcionasse com cache, seria necessário criar a camada com a closure.
    //let output_layer = Layer::new_with_function(128, 10, closure, Sigmoid::derivative);

    let mut last_epoch_weights = hidden_layer3.weights().clone();

    //network.add_layer(input_layer);
    network.add_layer(hidden_layer1);
    network.add_layer(hidden_layer2);
    network.add_layer(hidden_layer3);
    network.add_layer(output_layer);

    //while epoch < max_epochs {
    println!("Epoch {}.", epoch);
    println!("Training on EMNIST DataSet...");
    train_emnist(&mut network, training_samples, normalize_cast_f64);
    //batch_train_emnist(&mut network, training_samples);
    //println!("HL3 weights: {}", network.borrow_layer(3).weights());
    println!(
        "HL3 layer wieghts changed? {}",
        last_epoch_weights != *network.borrow_layer(3).weights()
    );
    last_epoch_weights = network.borrow_layer(3).weights().clone();

    println!("Test on Training samples...");
    test_emnist_on_training(&mut network, test_samples, normalize_cast_f64);
    println!("Testing neural net...");
    test_emnist(&mut network, test_samples, normalize_cast_f64);
    epoch += 1;


    let randomize_translation = |img:Vec<u8>|{
        
        let mut image_matrix = Matrix::from_vec(28, 28, normalize_cast_f64(img));
        
        let amount = rand::random_range(2..6); 
        match rand::random_range(0..4){
            0 => image_matrix.mut_translate_down(amount),    
            1 => image_matrix.mut_translate_up(amount),    
            2 => image_matrix.mut_translate_right(amount),    
            3 => image_matrix.mut_translate_left(amount),
            _ => print!("How?")    
        }
        image_matrix.data_mov()
    };

    println!("Epoch {}.", epoch);
    println!("Training on EMNIST DataSet With Translation...");
    train_emnist(&mut network, training_samples, randomize_translation);
    //batch_train_emnist(&mut network, training_samples);
    //println!("HL3 weights: {}", network.borrow_layer(3).weights());
    // println!(
    //     "HL3 layer wieghts changed? {}",
    //     last_epoch_weights != *network.borrow_layer(3).weights()
    // );
    // last_epoch_weights = network.borrow_layer(3).weights().clone();

    println!("Test on Training samples...");
    test_emnist_on_training(&mut network, test_samples, randomize_translation);
    println!("Testing neural net...");
    test_emnist(&mut network, test_samples, randomize_translation);
    //}
}

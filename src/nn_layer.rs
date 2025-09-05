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
    zed: Matrix,
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
            zed: Matrix::new(layer_neurons, 1),
            deltas: Matrix::new(layer_neurons, 1),
            weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
            biases: Matrix::new(layer_neurons, 1),
            activation: activation_function,
        }
    }

    pub fn neuron_qty(&self) -> usize {
        self.neurons.rows()
    }

    pub fn neurons(&self) -> &Matrix {
        &self.neurons
    }
    pub fn deltas(&self) -> &Matrix {
        &self.deltas
    }
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }
    pub fn zed(&self) -> &Matrix {
        &self.zed
    }

    /**
     * Fixa os valores dos pesos para geração de casos de teste
     */
    pub fn fix_weights(&mut self, weights: Matrix) {
        assert!(weights.rows() == self.weights.rows() && weights.cols() == self.weights.cols());
        self.weights = weights;
    }
    /**
     * Fixa os valores dos pesos para geração de casos de teste
     */
    pub fn fix_bias(&mut self, biases: Matrix) {
        assert!(biases.rows() == self.biases.rows() && biases.cols() == self.biases.cols());
        self.biases = biases;
    }
    /**
     * Fixa os valores da soma ponderada para geração de casos de teste
     */
    pub fn fix_zed(&mut self, zed: Matrix) {
        assert!(zed.rows() == self.zed.rows() && zed.cols() == self.zed.cols());
        self.zed = zed;
    }
    /**
     * Fixa os valores do erro para geração de casos de teste
     */
    pub fn fix_deltas(&mut self, deltas: Matrix) {
        assert!(deltas.rows() == self.deltas.rows() && deltas.cols() == self.deltas.cols());
        self.deltas = deltas;
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
            self.zed[i][0] = biased_values[i][0]; //z
            self.neurons[i][0] = self.activation.activate(biased_values[i][0]);
        }
    }

    pub fn cost(&mut self, expected: &Matrix) -> f64 {
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
     * Primeira implementação da retropropagação, para um único caso de testes,
     * seguindo as fórmulas da referência.
     * TO-DO: Importante! Estender Matrix e reimplementar essa função com operações matriciais
     *                              \--Produto Externo e Produto de Hadamard
     *
     * Visualização chave: https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0/
     * Fórmula:
     * ∂C/∂w = ∂z/∂w * ∂a/∂z * ∂C/∂a
     *
     * ∂z/∂w = a_(L-1).
     * ∂aL/∂z = activation'(z)
     * ∂C/∂a = 2(a - y)
     * */
    pub fn backpropagate_output_layer(
        &mut self,
        expected: &Matrix,
        prev_activations: &Matrix,
        cost_derivative: impl Fn(f64, f64) -> f64,
    ) -> Matrix {
        let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
        for i in 0..self.neurons.rows() {
            //∂aL/∂z = activation'(z) - Derivada parcial de a por z
            let a_zed_partial_derivative = self.activation.derivative(self.zed[i][0]);
            //∂C/∂a = 2(a - y) - Derivada parcial de C por a
            let c_a_partial_derivative = cost_derivative(self.neurons[i][0], expected[i][0]);
            //Original: let c_a_partial_derivative = 2.0 * (self.neurons[i][0] - expected[i][0]);

            //δ = hadamard_product(∂C/∂a, ∂aL/∂z).
            //Detalhe: o vetor delta é a derivada em função dos viéses ∂C/∂b = ∂z/∂b * ∂a/∂z * ∂C/∂a
            //já que ∂z/∂b = 1
            self.deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;
            for j in 0..self.weights.cols() {
                //∂C/∂w
                //∂z/∂w = a_(L-1).
                weight_derivatives[i][j] = prev_activations[j][0] * self.deltas[i][0];
            }
        }
        weight_derivatives
    }

    /**
     * Fórmulas:
     * 1 neuronio:
     * ∂C/∂a_(l-1) = ∂z/∂a_(l-1) * ∂a/∂z * ∂C/∂a
     *
     * ∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * ∂z_l/∂a_(l-1) * ∂a_l/∂z_l * ∂C/∂a_l
     * ∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * ∂z_l/∂a_(l-1) * δl
     *
     * n neuronios:
     * ∂C/∂a_(l-1) = sum(∂z/∂a_(l-1) * ∂a/∂z * ∂C/∂a)
     *
     * ∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * ∂a_l/∂z_l * ∂C/∂a_l)
     * ∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
     *
     * Definições finais:
     * ∂z_(l-1)/∂w_(l-1) = a_(L-1).
     * ∂a_(l-1)/∂z_(l-1) = activation'(z)
     * ∂C/∂a_(l-1) = sum(∂z/∂a_(l-1) * δl_l)
     * ∂z/∂a_(l-1) = w_l
     */
    pub fn backpropagate_hidden_layer(
        &mut self,
        next_layer: &Layer<T>,
        prev_activations: &Matrix,
    ) -> Matrix {
        let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
        //Transposição para que as dimensões estejam compatíveis.
        //Desnecessária pois wt[i][j] == w[j][i]
        //let weight_transpose = next_layer.weights.transpose();
        for i in 0..self.neurons.rows() {
            //∂aL/∂z = activation'(z) - Derivada parcial de a por z
            let a_zed_partial_derivative = self.activation.derivative(self.zed[i][0]);
            let mut c_a_partial_derivative = 0.0;
            //∂z/∂a_(l-1) * δ_l
            for j in 0..next_layer.weights.rows() {
                c_a_partial_derivative += next_layer.weights[j][i] * next_layer.deltas[j][0];
            }
            //δ = ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
            self.deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;

            for j in 0..self.weights.cols() {
                //∂z/∂w = a_(L-1).
                //∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
                //∂C/∂Cw_(l-1) = a_(L-1) * δ
                weight_derivatives[i][j] = prev_activations[j][0] * self.deltas[i][0];
            }
        }
        weight_derivatives
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

        let linear_transform = &layer2.weights * &layer1.weights;
        let bias_transform = (&layer2.weights * &layer1.biases) + &layer2.biases;

        print!("Neurons:{}", layer2.neurons);
        let expected = (linear_transform * input_mock) + &bias_transform;
        print!("Expected: {}", expected);

        assert!(expected == layer2.neurons);
    }

    #[test]
    fn test_backpropagate_output_layer() {
        println!("Back Propagate ---");
        let output_n = 3;
        let input_layer_n = 5;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut output_layer = Layer::new(input_layer_n, output_n, Relu {});
        let weights_mock = Matrix::from_vec(
            output_n,
            input_layer_n,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
        );
        let bias_mock = Matrix::from_vec(output_n, 1, vec![0.01, 0.02, 0.03]);

        output_layer.fix_bias(bias_mock);
        output_layer.fix_weights(weights_mock);

        let expected_mock = Matrix::from_vec(output_n, 1, vec![1.0, 1.0, 1.0]);
        let previous_mock = Matrix::from_vec(input_layer_n, 1, vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        output_layer.propagate(&previous_mock);

        //  Cálculo manual da matriz esperada ao fim da operação via Calculadora de Matrizes
        //[ -0,48                                   | -0.24 -0.24 -0.24 -0.24 -0.24 |
        //   2,04   X [0.5, 0.5, 0.5, 0.5, 0.5] ->  | 1.02   1.02   1.02    1.02    1.02 |
        //   4,56]                                  | 2.28   2.28   2.28    2.28    2.28 |
        println!("{},{},{}", (0 % 3), (1 % 3), (2 % 3));
        let expected_derivatives = Matrix::from_vec(
            3,
            5,
            vec![
                -0.24, -0.24, -0.24, -0.24, -0.24, 1.02, 1.02, 1.02, 1.02, 1.02, 2.28, 2.28, 2.28,
                2.28, 2.28,
            ],
        );

        let weight_derivatives = output_layer.backpropagate_output_layer(
            &expected_mock,
            &previous_mock,
            |a: f64, b: f64| 2.0 * (a - b),
        );
        // output_layer.backpropagate_output_layer(&expected_mock, &previous_mock);
        println!("Weight Derivatives:{}", weight_derivatives);
        println!("Expected Derivatives:{}", expected_derivatives);

        assert!(weight_derivatives == expected_derivatives);
    }

    #[test]
    fn test_backpropagate_hidden_layer() {
        println!("Back Propagate ---");
        let output_n = 3;
        let layer_n = 4;
        let input_layer_n = 2;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut hidden_layer = Layer::new(input_layer_n, layer_n, Relu {});
        let mut output_layer = Layer::new(layer_n, output_n, Relu {});
        let weights_mock = Matrix::from_vec(
            layer_n,
            input_layer_n,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        );
        let bias_mock = Matrix::from_vec(layer_n, 1, vec![0.01, 0.02, 0.03, 0.04]);
        let zed_mock = Matrix::from_vec(layer_n, 1, vec![0.5, -0.1, 0.8, -0.2]);

        hidden_layer.fix_bias(bias_mock);
        hidden_layer.fix_weights(weights_mock);
        hidden_layer.fix_zed(zed_mock);

        let output_weights_mock = Matrix::from_vec(
            output_n,
            layer_n,
            vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
        );
        let deltas_mock = Matrix::from_vec(output_n, 1, vec![0.9, -0.5, 0.2]);

        output_layer.fix_weights(output_weights_mock);
        output_layer.fix_deltas(deltas_mock);

        let previous_mock = Matrix::from_vec(input_layer_n, 1, vec![1.0, 0.5]);

        /*
                  Cálculo manual da matriz esperada ao fim da operação via Calculadora de Matrizes
                | 0.62 |                | 1.0 |   | 0.62 |                            | 0.62*1.0 0.62*0.5 | |0.62 0.31 |
        W^T * δ | 0.68 | had Relu'(zed) | 0.0 | = | 0    | ext. prev^T | 1.0  0.5 | = | 0         0       |=| 0     0  |
                | 0.74 |                | 1.0 |   | 0.74 |                            | 0.74*1.0 0.74*0.5 | |0.74 0.37 |
                | 0.8  |                | 0.0 |   | 0    |                            | 0         0       | | 0      0 |
                 */
        println!("{},{},{}", (0 % 3), (1 % 3), (2 % 3));
        let expected_derivatives =
            Matrix::from_vec(4, 2, vec![0.62, 0.31, 0.0, 0.0, 0.74, 0.37, 0.0, 0.0]);

        let weight_derivatives =
            hidden_layer.backpropagate_hidden_layer(&output_layer, &previous_mock);

        println!("Weight Derivatives:{}", weight_derivatives);
        println!("Expected Derivatives:{}", expected_derivatives);

        assert!(weight_derivatives == expected_derivatives);
    }
}

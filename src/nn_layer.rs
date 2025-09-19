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

pub struct Gradient {
    pub weight: Matrix,
    pub delta: Matrix,
}

pub trait ActivationFunction {
    fn activate(val: f64) -> f64;
    fn derivative(val: f64) -> f64;
}

pub struct Sigmoid {}
impl ActivationFunction for Sigmoid {
    fn activate(val: f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-val))
    }
    fn derivative(val: f64) -> f64 {
        let sigma = Sigmoid::activate(val);
        sigma * (1.0 - sigma)
    }
}
pub struct Relu {}
impl ActivationFunction for Relu {
    fn activate(val: f64) -> f64 {
        f64::max(0.0, val)
    }
    fn derivative(val: f64) -> f64 {
        match val {
            0.0 | _ if val < 0.0 => 0.0,
            _ => 1.0,
        }
    }
}
pub struct Identity {}
impl ActivationFunction for Identity {
    fn activate(val: f64) -> f64 {
        val
    }
    fn derivative(_: f64) -> f64 {
        1.0
    }
}

// type Link = Box<Layer>;
pub struct Layer {
    neurons: Matrix,
    zed: Matrix,
    weights: Matrix,
    biases: Matrix,
    activation_function: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64, 
}



impl Layer  {
    pub fn new<F:ActivationFunction>(
        prev_layer_neurons: usize,
        layer_neurons: usize,
    ) -> Layer {
        Layer {
            neurons: Matrix::new(layer_neurons, 1),
            zed: Matrix::new(layer_neurons, 1),
            weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
            biases: Matrix::new(layer_neurons, 1),
            activation_function: F::activate,
            activation_derivative: F::derivative,
        }
    }
    
    pub fn neuron_qty(&self) -> usize {
        self.neurons.rows()
    }

    pub fn neurons(&self) -> &Matrix {
        &self.neurons
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

    pub fn load_input(&mut self, input: Matrix) {
        self.neurons = input;
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
            self.neurons[i][0] = (self.activation_function)(biased_values[i][0]);
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
        cost_derivative: &dyn Fn(f64, f64) -> f64,
    ) -> Gradient {
        let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
        let mut deltas = Matrix::new(self.neuron_qty(), 1);

        for i in 0..self.neurons.rows() {
            //∂aL/∂z = activation'(z) - Derivada parcial de a por z
            let a_zed_partial_derivative = (self.activation_derivative)(self.zed[i][0]);
            //∂C/∂a = 2(a - y) - Derivada parcial de C por a
            let c_a_partial_derivative = cost_derivative(self.neurons[i][0], expected[i][0]);
            //Original: let c_a_partial_derivative = 2.0 * (self.neurons[i][0] - expected[i][0]);
            //δ = hadamard_product(∂C/∂a, ∂aL/∂z).
            //Detalhe: o vetor delta é a derivada em função dos viéses ∂C/∂b = ∂z/∂b * ∂a/∂z * ∂C/∂a
            //já que ∂z/∂b = 1
            deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;
            for j in 0..self.weights.cols() {
                //∂C/∂w
                //∂z/∂w = a_(L-1).
                weight_derivatives[i][j] = prev_activations[j][0] * deltas[i][0];
            }
        }
        Gradient {
            weight: weight_derivatives,
            delta: deltas,
        }
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
        next_layer_weights: &Matrix,
        next_layer_deltas: &Matrix,
        prev_activations: &Matrix,
    ) -> Gradient {
        let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
        let mut deltas = Matrix::new(self.neuron_qty(), 1);
        //Transposição para que as dimensões estejam compatíveis.
        //Desnecessária pois wt[i][j] == w[j][i]
        //let weight_transpose = next_layer.weights.transpose();
        for i in 0..self.neurons.rows() {
            //∂aL/∂z = activation'(z) - Derivada parcial de a por z
            let a_zed_partial_derivative = (self.activation_derivative)(self.zed[i][0]);
            let mut c_a_partial_derivative = 0.0;
            //∂z/∂a_(l-1) * δ_l
            for j in 0..next_layer_weights.rows() {
                c_a_partial_derivative += next_layer_weights[j][i] * next_layer_deltas[j][0];
            }
            //δ = ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
            deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;

            for j in 0..self.weights.cols() {
                //∂z/∂w = a_(L-1).
                //∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
                //∂C/∂Cw_(l-1) = a_(L-1) * δ
                weight_derivatives[i][j] = prev_activations[j][0] * deltas[i][0];
            }
        }
        Gradient {
            weight: weight_derivatives,
            delta: deltas,
        }
    }

    pub fn adjust_parameters(&mut self, gradients: Gradient, learning_rate: f64) {
        //Ajusta os pesos
        self.weights -= &gradients.weight.scalar_product(learning_rate);
        //Ajusta os viéses
        self.biases -= &gradients.delta.scalar_product(learning_rate);
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
        let layer2_n = 3;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut layer1 =
            Layer::new::<Identity>(input_n, layer1_n);
        let mut layer2 =
            Layer::new::<Identity>(layer1_n, layer2_n);
        let input_mock = Matrix::from_vec(input_n, 1, vec![1.0, 1.0, 1.0]);
        let weights1_mock = Matrix::from_vec(
            layer1_n,
            input_n,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
        );
        let weights2_mock = Matrix::from_vec(
            layer2_n,
            layer1_n,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
        );
        let bias1_mock = Matrix::from_vec(layer1_n, 1, vec![0.01, 0.02, 0.03, 0.04, 0.05]);
        let bias2_mock = Matrix::from_vec(layer2_n, 1, vec![0.01, 0.02, 0.03]);

        layer1.fix_bias(bias1_mock);
        layer1.fix_weights(weights1_mock);
        layer2.fix_bias(bias2_mock);
        layer2.fix_weights(weights2_mock);

        layer1.propagate(&input_mock);
        layer2.propagate(&layer1.neurons);

        print!("Neurons:{}", layer2.neurons);
        let expected =  Matrix::from_vec(3,1,vec![4.565, 10.65, 16.735]);
        print!("Expected: {}", expected);
        assert!(expected == layer2.neurons);
    }

    #[test]
    fn test_backpropagate_output_layer() {
        println!("Back Propagate ---");
        let output_n = 3;
        let input_layer_n = 5;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut output_layer = Layer::new::<Relu>(input_layer_n, output_n, );
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

        let gradient = output_layer.backpropagate_output_layer(
            &expected_mock,
            &previous_mock,
            &|a: f64, b: f64| 2.0 * (a - b),
        );
        // output_layer.backpropagate_output_layer(&expected_mock, &previous_mock);
        println!("Weight Derivatives:{}", gradient.weight);
        println!("Expected Derivatives:{}", expected_derivatives);

        assert!(gradient.weight == expected_derivatives);
    }

    #[test]
    fn test_backpropagate_hidden_layer() {
        println!("Back Propagate ---");
        let output_n = 3;
        let layer_n = 4;
        let input_layer_n = 2;
        //Camadas com pesos aleatórios e viéses inicializados em 0
        let mut hidden_layer = Layer::new::<Relu>(input_layer_n, layer_n);
        let mut output_layer = Layer::new::<Relu>(layer_n, output_n);
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

        let gradient = hidden_layer.backpropagate_hidden_layer(
            &output_layer.weights,
            &deltas_mock,
            &previous_mock,
        );

        println!("Weight Derivatives:{}", gradient.weight);
        println!("Expected Derivatives:{}", expected_derivatives);

        assert!(gradient.weight == expected_derivatives);
    }
}

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
use rand;
use rand::Rng;
use std::cmp;
use std::collections;
use std::fmt;
use std::ops::AddAssign;
use std::ops::MulAssign;
use std::ops::SubAssign;
use std::ops::{Add, Index, IndexMut, Mul}; //Traits para o operador de índice []

const FMT_NUM_WIDTH: usize = 8;
const FMT_NUM_PRECISION: usize = 3;
const EPSILON: f64 = 1e-9; //Usado para implementar comparação absoluta de floats. Valor obtido

#[derive(Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

/**  Implementação de struct para encapsular as operações matriciais
 Motivação: abstrair a implementação formal da álgebra linear usada na rede neural,
 possibilitando a implementação de novas structs com otimização nas operações.
 Operações como o produto escalar ou a adição de matrizes são exemplos clássicos de paralelismo
 de dados. Uma nova implementação da struct Matrix pode utilizar técnicas de programação
 paralela para otimizar a execução dessas operações.
*/
impl Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn data(&self) -> &Vec<f64>{
        &self.data
    }

    pub fn new(num_rows: usize, num_cols: usize) -> Matrix {
        Matrix {
            rows: num_rows,
            cols: num_cols,
            data: vec![0.0; num_rows * num_cols], //Dados inicializados com 0.0
        }
    }

    pub fn from_vec(num_rows: usize, num_cols: usize, _data: Vec<f64>) -> Matrix {
        Matrix {
            rows: num_rows,
            cols: num_cols,
            data: _data, //Dados inicializados com 0.0
        }
    }
    /*Intervalo Glorot: média 0 e variância 2/n_in+n_out
        [-(sqrt(6)/sqrt(n_in+n_out)),(sqrt(6)/sqrt(n_in+n_out))]
    */
    pub fn new_random_glorot(n_in: usize, n_out: usize) -> Matrix {
        let high = ((6.0 as f64).sqrt() / ((n_in + n_out) as f64)).sqrt();
        let low = -high;
        let mut rng = rand::rng();
        let mut distribution = rand::distr::Uniform::new(low, high).unwrap();
        Matrix {
            rows: n_in,
            cols: n_out,
            data: (0..(n_in * n_out))
                .map(|_| rng.sample(&distribution))
                .collect(),
        }
    }
    /*//Intervalo He: média 0 e variância sqrt(2/n_in)
        [-2sqrt(2/n_in), 2sqrt(2/n_in)]
    */
    pub fn new_random_he(n_in: usize, n_out: usize) -> Matrix {
        let std_deviation = (2.0 / n_in as f64).sqrt();
        let mut rng = rand::rng();
        let distribution = rand_distr::Normal::new(0.0, std_deviation).unwrap();

        Matrix {
            rows: n_in,
            cols: n_out,
            data: (0..(n_in * n_out))
                .map(|_| rng.sample(&distribution))
                .collect(),
        }
    }
    /**
     * Inicializa uma nova matriz com valores aleatórios.
     * Por padrão, usa a inicialização He voltada à ativação via ReLU
     */
    pub fn new_random(num_rows: usize, num_cols: usize) -> Matrix {
        Matrix::new_random_he(num_rows, num_cols)
    }

    //Função auxiliar para calcular o índice do vetor de dados com base nos índices de linha e coluna
    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows); //Índice da linha deve ser válido 0 <= row < self.rows
        assert!(col < self.cols); //Índice da coluna deve ser válido 0 <= row < self.rows
        return row * self.cols + col;
    }

    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }

    //Implementação simples do produto de 2 matrizes
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        //Para produto de matrizes,
        // se a primeira tem dimensões      M x N
        // a segunda deve possuir dimensões N x O
        assert!(self.cols == other.rows);
        let mut product = Matrix::new(self.rows, other.cols);
        //product [i,j] = sum(self[i][0]..[i][n] * other[0][j]..[n][j])
        for i in 0..product.rows {
            for j in 0..other.cols {
                //Da notação matemática: prod[i,j] = sum(self[i,k] * other[k,j])
                let mut row_product = 0.0;
                for k in 0..self.cols {
                    row_product += self[i][k] * other[k][j];
                }
                product[i][j] = row_product;
            }
        }
        return product;
    }

    /* Implementação ingênua da transposição de matrizes.
      TO-DO: Considerar alternativas à ineficiência de alocar nova struct
      TO-DO: Verificar como aproveitar a representação em vetor para otimizar a transposição
    */
    pub fn transpose(&self) -> Matrix {
        let mut transpose = Matrix {
            rows: (self.cols),
            cols: (self.rows),
            data: (vec![0.0; self.cols * self.rows]),
        };
        //T_ij = A_ji
        for i in 0..transpose.rows {
            for j in 0..transpose.cols {
                transpose[i][j] = self[j][i];
            }
        }
        return transpose;
    }

    pub fn hadamard_product(&self, other: &Matrix) -> Matrix {
        assert!(self.rows == other.rows && self.cols == other.cols);
        let mut productVec = vec![0.0; self.num_elements()];
        for i in 0..self.num_elements() {
            productVec[i] = self.data[i] * other.data[i];
        }
        Matrix {
            rows: self.rows(),
            cols: self.cols,
            data: productVec,
        }
    }
    pub fn mut_hadamard_product(&mut self, other: &Matrix) {
        if (self.rows == other.rows && self.cols == other.cols) {
            for i in 0..self.num_elements() {
                self.data[i] = self.data[i] * other.data[i];
            }
            return;
        }
        //Broadcasting simplificado quando other é matriz coluna
        assert!(self.rows == other.rows && other.cols == 1);
        for i in 0..self.num_elements() {
            let j = i % other.rows;
            self.data[i] = self.data[i] * other.data[j];
        }
    }

    pub fn scalar_product(self, scalar: f64) -> Matrix {
        let mut result = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: vec![0.0; self.num_elements()],
        };
        for i in 0..self.num_elements() {
            result.data[i] = self.data[i] * scalar;
        }
        result
    }
    
    pub fn mut_scalar_product(&mut self, scalar: f64) {
        for i in 0..self.num_elements() {
            self.data[i] = self.data[i] * scalar;
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} x {})\n", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "| ")?;
            let start_index = i * self.cols;
            let slice = &self.data[start_index..start_index + self.cols];
            for i in 0..self.cols {
                //Imprime alinhado à direita (>) com largura FMT_NUM_WIDTH e FMT_NUM_PRECISION casas decimais
                write!(f, "{:>FMT_NUM_WIDTH$.FMT_NUM_PRECISION$}  ", slice[i])?;
            }
            write!(f, "|\n")?;
        }
        write!(f, "")
    }
}

//Implementação de índices em formato de tupla no estilo matriz[(linha,coluna)]
impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;

        return &self.data[self.index(row, col)];
    }
}

//Implementação de índices através de tupla: a[(linha,coluna)]
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let idx = self.index(row, col); //necessário separar por conta do borrow abaixo
        return &mut self.data[idx];
    }
}

//Implementação de índices em sintaxe padrão para vetores bidimensionais matriz[linha][coluna]
impl Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        let idx_base = self.index(index, 0);
        let slice = &self.data[idx_base..idx_base + self.cols];
        return slice;
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx_base = index * self.cols;
        let slice = &mut self.data[idx_base..idx_base + self.cols];
        return slice;
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        //Comparação de floats com == pode gerar problemas.
        //Método de comparação absoluta, suficiente enquanto só usamos para testes
        //Uma biblioteca completa implementaria métodos mais robustos: https://floating-point-gui.de/errors/comparison/
        for i in 00..self.num_elements() {
            if (self.data[i] - other.data[i]).abs() > EPSILON {
                return false;
            }
        }
        true
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}
impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(rhs)
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.num_elements()) {
            result.data[i] = self.data[i] + rhs.data[i];
        }
        return result;
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        for i in 0..(self.num_elements()) {
            self.data[i] = self.data[i] + rhs.data[i];
        }
    }
}
impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        for i in 0..(self.num_elements()) {
            self.data[i] = self.data[i] - rhs.data[i];
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope. (Rust Book)
    use super::*;
    #[test]
    fn test_indexes() {
        let mut base_matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, -4.0, 0.0, 5.0],
        };
        print!("{}\n", base_matrix);
        base_matrix[0][1] = 5.0;
        print!("{}\n", base_matrix);
        base_matrix[(1, 2)] = -40.0;
        print!("{}\n", base_matrix);
    }

    #[test]
    fn test_mult() {
        let base_matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, -4.0, 0.0, 5.0],
        };
        let other = Matrix {
            rows: 3,
            cols: 2,
            data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],
        };
        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![12.0, 10.0, 12.0, -23.0],
        };
        print!("Base: {}", base_matrix);
        print!("Mult: {}", other);

        let result = base_matrix * other;

        print!("Result: {}", result);
        assert!(expected == result);
    }
    #[test]
    #[should_panic]
    fn test_invalid_mult() {
        let base_matrix = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let other = Matrix {
            rows: 3,
            cols: 2,
            data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],
        };
        let panic = other * base_matrix;
    }
    #[test]
    fn test_transpose() {
        let base_matrix = Matrix {
            rows: 3,
            cols: 4,
            data: vec![
                5.0, 8.0, -1.0, 7.0, 0.0, -2.0, 6.0, 1.0, 4.0, 3.0, 9.0, -5.0,
            ],
        };
        let transpose = Matrix {
            rows: 4,
            cols: 3,
            data: vec![
                5.0, 0.0, 4.0, 8.0, -2.0, 3.0, -1.0, 6.0, 9.0, 7.0, 1.0, -5.0,
            ],
        };
        print!("Base: {}", base_matrix);
        let transpose_matrix = base_matrix.transpose();
        print!("Transposed: {}", transpose_matrix);

        assert!(transpose == transpose_matrix);
    }

    #[test]
    fn test_add() {
        let base_matrix = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, -4.0],
        };
        let other = Matrix {
            rows: 2,
            cols: 2,
            data: vec![2.0, 7.0, -1.0, 0.0],
        };
        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![3.0, 9.0, 2.0, -4.0],
        };
        let result = base_matrix + &other;
        assert!(expected == result);
    }
    #[test]
    #[should_panic]
    fn test_invalid_add() {
        let base_matrix = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let other = Matrix {
            rows: 3,
            cols: 2,
            data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],
        };
        let panic = other + &base_matrix;
    }
}

# Implementando redes neurais em Rust: álgebra

Material de referência principal: https://www.3blue1brown.com/topics/neural-networks
Referência de Rust: https://doc.rust-lang.org/book/

Como usei a IA para auxiliar?
Iniciei instruindo o bot a não exibir exemplos de código da implementação, visto que os exemplos invalidariam o propósito desse exercício. A maior vantagem da IA foi na criação de exemplos para gerar os testes de validação da implementação: 
"Não exiba exemplos de código implementando as funções." 
"Sabendo a média e a variância, quais passos são utilizados para fazer a amostragem do intervalo?."
"Apresente 2 matrizes e suas transpostas. Use matrizes não quadradas com dimensões entre 3 e 5"

Por conta da mudança de emprego, e de medicação, me encontro estudando os conceitos por trás dos modelos de Inteligência Artificial. Como eu não sou um caso especial, também me interesso pelo que está em alta no momento: Grandes Modelos de Linguagem como ChatGPT. Para entender esses modelos, é necessário compreender o mecanismo básico que deu origem a estes monstros: Redes Neurais.

Como fonte original eu revisitei a sequência de vídeos do canal [3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), com transcrição disponível no [link](https://www.3blue1brown.com/topics/neural-networks). As visualizações incluídas nos vídeos são muito bem feitas e passam uma compreensão intuitiva da matemática por trás disso tudo. Mas ao final de tudo eu ainda fiquei com uma pulga atrás da orelha. Como eu garanto que realmente entendi isso tudo?

A resposta é bem óbvia: implementando tudo isso do zero.
E por que fazer isso em Rust? Porque é uma linguagem que eu quero aprender.

Resolvi documentar todo esse processo em uma sequência de artigos demonstrando o avanço dessa implementação, usando um processo *bottom-up*. E se começamos debaixo, precisamos implementar a base matemática para a maior parte das operações: álgebra de matrizes. 

## Representando matrizes

Tudo começa com a escolha da representação das matrizes (no conceito matemático) em nossa base de código. Matematicamente, uma matriz representa um conjunto de elementos através de 2 dimensões: linhas e colunas. Nos referimos a um elemento específico através de sua posição na matriz relativa a cada dimensão: o elemento a_1,2 encontra-se na linha 1 e coluna 2.

Na programação, uma matriz pode ser representada com tipos primitivos da linguagem: um array de arrays `matriz:[[f64]]`, ou como um vetor de vetores `Vec<Vec<f64>>`. Ambas as alternativas permitiriam o acesso a matriz na forma usual `matriz[linha][coluna]`.

Outra maneira de representar uma matriz é com a criação de uma estrutura de dados que agrupe as propriedades essenciais deste objeto matemático. Em Rust, agrupamos propriedades utilizando `structs`:
```
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>, //Vetor de floats 64bits
}
``` 
E por que criamos uma struct ao invés de usar tipos primitivos da linguagem? 

A grande vantagem de utilizar uma struct é a flexibilidade na forma de armazenar os elementos da matriz. Se os elementos estão "empacotados" em uma estrutura opaca, podemos modificar a implementação interna sem afetar o código que depende das matrizes. Por enquanto não estou interessado em performance, mas uma implementação efetiva de redes neurais focaria em implementações eficientes da manipulação dos dados de matrizes. Encapsular nossa implementação nos permitiria evoluir esse código com maior facilidade no futuro. 

Um detalhe essencial na nossa implementação é o armazenamento dos dados de nossa matriz em um vetor unidimensional `data: Vec<f64>`. Esse é um truque clássico da computação para  otimizar o acesso de memória em operações que percorrem a matriz sequencialmente. Nossa representação planifica a matriz, agrupando as linhas lado a lado. Visualmente, fazemos a transformação dessa forma:

//https://latex2image.joeraut.com/
\begin{bmatrix} a_{00} & a_{01} & a_{02} \\ a_{10} & a_{11} & a_{12} \\ a_{20} & a_{21} & a_{22} \end{bmatrix} \to \begin{bmatrix} a_{00} & a_{01} & a_{02} & a_{10} & a_{11} & a_{12} & a_{20} & a_{21} & a_{22} \end{bmatrix}

[00 01 02
 10 11 12         ->    [00 01 02 10 11 12 20 21 22]
 20 21 22 ]

Com essa representação, como acessamos um elemento específico usando os índices de linha e coluna? O algoritmo é simples:
1. A posição do primeiro elemento da linha depende do número de colunas. Se a primeira linha inicia no índice 0, a segunda linha vai iniciar no índice 0 + num_cols, a terceira no índice 0 + 2num_cols e assim sucessivamente. Como indexamos partindo do zero, a sequência é reduzida à (linha * num_cols);
2. Sabendo o primeiro elemento da linha, o elemento desejado é alcançado avançando o número de colunas desejadas.

Portanto, o índice do elemento [i][j] no vetor de dados é dado por ( i * num_cols) + j. Na linguagem Rust, podemos definir o operador [] para nossa struct Matriz, permitindo que usemos instâncias dessa struct  com a sintaxe clássica das matrizes em computação. Isso é feito pela implementação do `trait` `Index<T>`. Implementamos a indexação da matriz da seguinte maneira:

```
impl Matrix {
    //Função auxiliar para calcular o índice do vetor de dados com base nos índices de linha e coluna
    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows); //Índice da linha deve ser válido 0 < row < self.rows
        assert!(col < self.cols); //Índice da coluna deve ser válido 0 < col < self.cols
        return row * self.cols + col;
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
        let idx_base = index * self.cols;                         // Índice do primeiro elemento da linha
        let slice = &self.data[idx_base..(idx_base + self.cols)]; // Slice do vetor contendo os dados da linha
        return slice;
    }
}
impl IndexMut<usize> for Matrix {...}          //implementação para elementos mutáveis
```

(colocar como anexo) Pensando ainda na conveniência de uso da struct, implementamos um método de formatação que me permita visualizar a estrutura da matriz criada:
```
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
```

Hora dos testes! Em Rust, os testes são especificados em um módulo de testes:
```
#[cfg(test)]
mod tests {
    use super::*; // importa nomes do módulo externo (Rust Book) 
    #[test]
    fn test_indexes() {
        //Matriz 2x3
        let mut base_matrix = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, -4.0, 0.0, 5.0], };
        print!("{}\n", base_matrix);
        base_matrix[0][1] = 5.0;
        print!("{}\n", base_matrix);
        base_matrix[(1,2)]=-40.0;
        print!("{}\n", base_matrix);
    }
}
``` 
Perfeito, temos uma matriz acessível através de índices: 
```
cargo test -- --nocapture
(2 x 3)
|  1.000  2.000  3.000 |
| -4.000  0.000  5.000 |
(2 x 3)
|    1.000     5.000     3.000 |
|   -4.000     0.000     5.000 |
(2 x 3)
|    1.000     5.000     3.000 |
|   -4.000     0.000   -40.000 |
```

Com nossa representação especificada, podemos iniciar a implementação das operações algébricas que precisamos para a rede neural.
Começamos pelo simples: multiplicação e adição de matrizes. 

A Multiplicação de matrizes é definida por 2 regras: 
1. Para multiplicar as matrizes A_mn e B_op, o número de colunas (n) da matriz A **deve** ser igual ao número de linhas (o) da matriz B;
2. O resultado da multiplicação é uma matriz C_mp (o número de linhas de A e o número de colunas de B), onde cada elemento C_ij é definido pela soma do produto dos elementos da linha i de A pela coluna j de B. Matematicamente: C_ij = sum_k(A_ik * B_kj);

O algoritmo que implementa esses passos da multiplicação de matrizes é mostrado abaixo. A implementação do *trait* Mult nos permite a sobrecarga do operador * para objetos da struct Matrix: 
```
impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        //Para produto de matrizes,
        // se a primeira tem dimensões      M x N
        // a segunda deve possuir dimensões N x O
        assert!(self.cols == rhs.rows);
        let mut product = Matrix::new(self.rows, rhs.cols);
        //product [i,j] = sum(self[i,0]..[i,n] * rhs[0,j]..[n,j])
        for i in 0..product.rows {      //product.rows == self.rows
            for j in 0..product.cols {  //product.cols == rhs.cols
                //Da notação matemática: prod[i,j] = sum(self[i,k] * rhs[k,j])
                let mut row_product = 0.0;
                for k in 0..self.cols {
                    row_product += self[i][k] * rhs[k][j];
                }
                product[i][j] = row_product;
            }
        }
        return product;
    }
}
```

Para conferir nossa implementação, geramos os testes da multiplicação:
```
    #[test]
    fn test_mult() {
        let base_matrix = Matrix {rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, -4.0, 0.0, 5.0],};
        let other = Matrix {rows: 3, cols: 2, data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],};
        let expected = Matrix {rows: 2, cols: 2, data: vec![12.0, 10.0, 12.0, -23.0],};
        let result = base_matrix * other;
        assert!(expected == result);
    }
    #[test]
    #[should_panic] //Testa se a função gera erro pela condição inválida
    fn test_invalid_mult() {
        let base_matrix = Matrix {rows: 1, cols: 3, data: vec![1.0, 2.0, 3.0],};
        let other = Matrix {rows: 3, cols: 2, data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],};
        let panic = other * base_matrix;
    }
```

A adição possui 2 regras:
1. A operação de adição só é definida para matrizes de mesmo tamanho
2. O resultado é uma matriz C, com elementos definidos por C_ij = A_ij + B_ij

A implementação inicial da adição dá-se sobrecarregando o operador + através do trait `Add`:

```
impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.data.len()) {
            result.data[i] = self.data[i] + rhs.data[i];
        }
        return result;
    }
}
```

Essa implementação ilustra uma das vantagens do uso de uma estrutura planificada para armazenar os dados da matriz: podemos aplicar a adição percorrendo ambos os vetores sequencialmente. Observe também que, em uma aplicação que exige performance, poderíamos paralelizar a operação "cortando" o vetor em múltiplos pedaços e aplicando a adição em cada pedaço simultaneamente. 

Implementamos também os testes para a adição usando matrizes de exemplo:
```
    #[test]
    fn test_add() {
        let base_matrix = Matrix {rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, -4.0],};
        let other = Matrix {rows: 2, cols: 2, data: vec![2.0, 7.0, -1.0, 0.0],};
        let expected = Matrix {rows: 2, cols: 2, data: vec![3.0, 9.0, 2.0, -4.0],};
        let result = base_matrix + &other;
        assert!(expected == result);
    }
    #[test]
    #[should_panic]
    fn test_invalid_add() {
        let base_matrix = Matrix {rows: 1,cols: 3,data: vec![1.0, 2.0, 3.0],};
        let other = Matrix {rows: 3,cols: 2,data: vec![2.0, 7.0, -1.0, 0.0, 4.0, 1.0],};
        let panic = other + &base_matrix;
    }
```

Observação: Os testes dependem da implementação do trait de comparação parcial, que sobrecarrega o operador (==).
```
impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        //Vec já implementa a comparação elemento a elemento, portanto não precisamos nos preocupar
        self.data == other.data
    }
```

A última operação matricial básica que precisamos implementar é a transposição de matrizes. Essa operação inverte as linhas e colunas de uma matriz:
1. A transposta da matriz A_mn é a matriz AT_nm;
2. Os elementos são definidos da forma AT_ij = A_ji;

Novamente, uma implementação direta é simples com os operadores já implementados:
```
    pub fn transpose(&self) -> Matrix {
        let mut transpose = Matrix {rows: (self.cols),cols: (self.rows),data: (vec![0.0; self.cols * self.rows]),};
        //T_ij = A_ji
        for i in 0..transpose.rows {
            for j in 0..transpose.cols {
                transpose[i][j] = self[j][i];
            }
        }
        return transpose;
    }
...
    #[test]
    fn test_transpose() {
        let base_matrix = Matrix {
            rows: 3,cols: 4,data: vec![5.0, 8.0, -1.0, 7.0, 0.0, -2.0, 6.0, 1.0, 4.0, 3.0, 9.0, -5.0,],
        };
        let transpose = Matrix {
            rows: 4,cols: 3,data: vec![5.0, 0.0, 4.0, 8.0, -2.0, 3.0, -1.0, 6.0, 9.0, 7.0, 1.0, -5.0,],
        };
        let transpose_matrix = base_matrix.transpose();
        assert!(transpose == transpose_matrix);
    }
```

Para finalizar nossa implementação de matrizes, só nos resta disponibilizar métodos para a inicialização de objetos desse tipo pela rede neural. Nossa rede vai usar dois tipos de matrizes:
1. Matrizes para o valor dos dados obtidos pela ativação dos neurônios: podem ser inicializadas em 0;
2. Matrizes para os pesos e viéses: devem ser inicializadas com valores aleatórios seguindo uma distribuição de probabilidade.

A primeira inicialização é muito simples:
```
    pub fn new(num_rows: usize, num_cols: usize) -> Matrix {
        Matrix { rows: num_rows, cols: num_cols,
            data: vec![0.0; num_rows * num_cols], //Dados inicializados com 0.0
        }
    }
```

Já a inicialização via distribuições de probabilidade exige a implementação da amostragem de dados. Embora a ideia seja implementar tudo "do zero", uma implementação adequada de amostragem aleatória está além do escopo, portanto implementamos essa inicialização com o uso das bibliotecas *rand* (*crate* em Rust). 

A distribuição de probabilidade apropriada para a rede neural depende da função de ativação escolhida. Entre as funções mais comuns estão a função sigmoide e ReLU. A primeira é beneficiada pela inicialização de Gloirot, enquanto a segunda usa o modo de inicialização He. A distribuição de probabilidade utilizada por estes métodos é dada a seguir:
1. Glorot: Usa o intervalo de dados [-sqrt(6/n_in+n_out),sqrt(6/n_in+n_out)], onde n_in é o número de neurônios da camada de entrada e n_out é o número de neurônios da camada atual. A amostragem é feita em uma distribuição uniforme dos valores desse intervalo;
2. He: Intervalo com desvio padrão sqrt(2/n_in) e média 0. A amostragem é feita de uma distribução Gaussiana (Normal).

```
    pub fn new_random_glorot(n_in: usize, n_out: usize) -> Matrix {
        let high = ((6.0 as f64).sqrt() / ((n_in + n_out) as f64)).sqrt();
        let low = -high;
        let mut rng = rand::rng();
        let mut distribution = rand::distr::Uniform::new(low, high).unwrap();
        Matrix { rows: n_in, cols: n_out,
            data: (0..(n_in * n_out))
                .map(|_| rng.sample(&distribution))
                .collect(),
        }
    }
    pub fn new_random_he(n_in: usize, n_out: usize) -> Matrix {
        let std_deviation = (2.0 / n_in as f64).sqrt();
        let mut rng = rand::rng();
        let distribution = rand_distr::Normal::new(0.0, std_deviation).unwrap();
        Matrix { rows: n_in, cols: n_out,
            data: (0..(n_in * n_out))
                .map(|_| rng.sample(&distribution))
                .collect(),
        }
    }
```

E assim concluímos nossa implementação fundamental de matrizes para as redes neurais. No próximo artigo iniciamos efetivamente a implementação da rede neural, implementando camadas e propagação. 

Quais são os próximos passos nessa implementação?
1. Todas as operações fundamentais podem ser otimizadas. Como as matrizes são essenciais para muitas áreas da computação, a literatura referente à otimização dessas operações é muito farta;
2. Implementar operações mutáveis. Todas as operações implementadas aqui geram uma nova matriz como resultado, alocando um novo objeto a cada iteração. Considerando o fluxo de operações das redes neurais, podemos otimizar o uso de memória através da manipulação de uma única matriz mutável para representar os dados de treinamento e validação;
3. Especificar a serialização das matrizes, permitindo que os valores sejam gravados e restaurados posteriormente.



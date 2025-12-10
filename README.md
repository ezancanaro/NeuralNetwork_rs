# NeuralNetwork_rs

Implementação de uma rede neural simples na linguagem Rust. O projeto foi criado para explorar os conceitos de redes neurais e o paradigma de programação incentivado pela linguagem. Não há nenhuma pretensão de transformar este projeto em uma implementação geral para redes neurais.

A implementação do projeto foi relatada em uma série de artigos publicados no Medium:

[Parte 1](https://medium.com/@ericzancanaro/redes-neurais-do-zero-em-rust-parte-1-matrizes-d22bf040a6d3)
[Parte 2](https://medium.com/@ericzancanaro/redes-neurais-do-zero-em-rust-parte-2-neurônios-e-propagação-94969aada9b5)
[Parte 3](https://medium.com/@ericzancanaro/redes-neurais-do-zero-em-rust-parte-3-retropropagação-2ff48d72bf2e)
[Parte 4](https://medium.com/@ericzancanaro/redes-neurais-do-zero-em-rust-parte-4-reconhecendo-números-manuscritos-37f82ab9240a)

Este código é publicado de maneira *open-source* e não apresenta garantias de manutenção ou funcionamento futuro

## Estrutura do Código

A implementação teve por objetivo explorar os conceitos matemáticos por trás das redes neurais. Para isso, implementamos as operações matriciais de forma manual.
O projeto tem a seguinte estrutura:

```
NeuralNetwork_rs
|__artigos           -- Rascunhos do texto dos artigos
|__src
|   |__nn_matrix.rs  -- Implementação da representação das matrizes e suas operações matemáticas (seriam tensores se fôssemos mais corretos)
|   |__nn_layer.rs   -- Estrutura das camadas de redes neurais, contendo os neurônios, pesos, vieses e as implementações da propagação e retropropagação
|   |__nn_network.rs -- Generalização da rede neural. Armazena as camadas e implementa as rotinas de treinamento, ajuste de parâmetros e classificação
|   |__nn_emnist.rs  -- Parser para os arquivos do dataset emnist, no formato binário do dataset MNIST original
|   |__nn_main.rs    -- Classe principal, implementa o treinamento e classificação do dataset emnist
|__target            -- Diretório com artefatos da compilação, gerado automaticamente pelo compilador
```
## Dependências e Compilação

O projeto pode ser construído com o toolset padrão da linguagem Rust, disponível no [site oficial](https://rust-lang.org/learn/get-started/). 

Dependências de projetos Rust são especificadas no arquivo `cargo.toml`. O projeto depende apenas das bibliotecas *rand* para a inicialização aleatória dos pesos.

```
[dependencies]
rand = "0.9.2"
rand_distr = "0.5.1"
```

Com o conjunto de ferramentas da linguagem configurado apropriadamente, o projeto pode ser executado, em modo debug, com o comando `cargo run`.
Para compilar a versão otimizada, utilize o comando `cargo build --release`.

## Identificação do Dataset EMNIST

Os arquivos contendo as imagens de treinamento e validação do dataset EMNIST podem ser encontradas no site do projeto:
[EMINST](https://www.nist.gov/itl/products-and-services/emnist-dataset)

A versão atual implementa um parser para a versão binária dos arquivos, não sendo compatível com o formato Matlab.
O caminho dos arquivos do dataset é fixo nas constantes definidas no arquivo `main.rs`

```
const EMNINST_TRAIN_IMAGES:&str = "emnist/emnist-digits-train-images-idx3-ubyte";
const EMNINST_TRAIN_LABELS:&str = "emnist/emnist-digits-train-labels-idx1-ubyte";
const EMNINST_TEST_IMAGES:&str = "emnist/emnist-digits-test-labels-idx1-ubyte";
const EMNINST_TEST_LABELS:&str = "emnist/emnist-digits-test-images-idx3-ubyte";
```

Para executar a validação do dataset, ajuste o caminho dos arquivos e compile com `cargo build --release`.



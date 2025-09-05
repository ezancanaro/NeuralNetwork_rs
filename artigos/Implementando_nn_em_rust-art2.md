Agora que temos a base para as operações matriciais podemos iniciar a implementação da nossa rede neural em si.
Conceitualmente, uma rede neural é construída através da conexão das camadas de entrada e saída, intermediadas por um conjunto de camadas ocultas.
Uma camada da rede neural é definida por:
1. Conjunto de neurônios: são os elementos que efetivamente processam os dados na rede neural. Os neurônios recebem um conjunto de N sinais de entrada e produzem 1 valor de saída. O número de sinais recebidos depende da quantidade de neurônios da camada anterior;
2. Função de ativação: de forma geral, são funções aplicadas ao valor de ativação do neurônio para inserir não-linearidade na rede. Tecnicamente a função de ativação pode ser definida para cada neurônio individual. Na prática, muitas arquiteturas de redes neurais utiliza a mesma função de ativação para todos os neurônios. Cada camada pode utilizar uma função de ativação própria, projetada para sua função na rede.
3. Conexões de neurônios: toda camada é conectada com a próxima através de valores que ponderam o efeito da camada anterior em cada um de seus neurônios. No modelo de camadas totalmente conectadas, cada neurônio da camada possui 1 conexão com **todos** os neurônios da camada anterior. As conexões são definidas por 2 valores:
   1. Pesos: representam o quão importante a conexão é para o valor de ativação do neurônio a ele conectado. Quanto maior o peso, maior o efeito do sinal do neurônio de origem no cálculo de valor do neurônio alvo; 
   2. Viéses: são valores adicionados ao valor resultante da soma ponderada dos pesos. Podemos entender que o viés desloca o resultado da operação à direita (se positivo) ou à esquerda (se negativo), criando uma nova dimensão para os ajustes da rede neural.


Com essa definição em mãos, podemos iniciar a nossa implementação em Rust. Primeiro de tudo, precisamos de um tipo que represente nossas funções de ativação. Pensando em outras linguagens que conheço, usaríamos o conceito de Interface para criar um tipo genérico que poderia ser implementado futuramente. Em Rust, podemos implementar um conceito análogo através da especificação de um trait:
```
pub trait ActivationFunction {
    fn activate(&self, val: f64) -> f64;
    fn derivative(&self, val: f64) -> f64;
}
```

Nosso trait especifica que qualquer struct que o implemente deve implementar 2 funções:
1. fn activate é a função de ativação em si, usada durante a propagação dos valores na rede neural. Ela deve receber um único float, que é o valor da soma ponderada do produto dos neurônios de entrada pelos pesos;
2. fn derivative é a derivada da função de ativação. Essa derivada será essencial quando implementarmos a retropropagação para permitir o aprendizado em nossa rede neural.

Sabendo como especificar nossas funções de ativação, podemos definir a struct de camadas: 
```
pub struct Layer {
    neurons: Matrix,     // Valores de ativação dos neurônios da camada
    weights: Matrix,     // Pesos
    biases: Matrix,      // Viéses
    activation: ActivationFunction, 
}
impl Layer {
    //Cria uma nova camada
    pub fn new(
        prev_layer_neurons: usize,
        layer_neurons: usize,
        activation_function: ActivationFunction, 
    ) -> Layer {
        Layer {
            neurons: Matrix::new(layer_neurons, 1),
            weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
            biases: Matrix::new_random(layer_neurons, 1),
            activation: activation_function,
        }
    }
}

```

Exceto que essa a sintaxe não funciona. Temos erros em 2 pontos: na definição do tipo da propriedade activation e na definição do tipo de nosso objeto no construtor. Pela primeira temos motivo para demonstrar as mensagens de erro do compilador Rust:

error[E0782]: expected a type, found a trait
  --> src\nn_layer.rs:54:17
   |
54 |     activation: ActivationFunction,
   |                 ^^^^^^^^^^^^^^^^^^
   |
help: you can add the `dyn` keyword if you want a trait object
   |
54 |     activation: dyn ActivationFunction,
   |                 +++

error[E0782]: expected a type, found a trait
  --> src\nn_layer.rs:62:30
   |
62 |         activation_function: ActivationFunction,
   |                              ^^^^^^^^^^^^^^^^^^
   |
help: use a new generic type parameter, constrained by `ActivationFunction`
   |
59 ~     pub fn new<T: ActivationFunction>(
60 |         prev_layer_neurons: usize,
61 |         layer_neurons: usize,
62 ~         activation_function: T,
   |
help: you can also use an opaque type, but users won't be able to specify the type parameter when calling the `fn`, having to rely exclusively on type inference
   |
62 |         activation_function: impl ActivationFunction,
   |                              ++++
help: alternatively, use a trait object to accept any type that implements `ActivationFunction`, accessing its methods at runtime using dynamic dispatch
   |
62 |         activation_function: &dyn ActivationFunction,
   |                              ++++

O que é mais interessante nessa ferramenta é o nível de informações fornecidas pelo compilador. Veja que, embora o erro encontrado seja o mesmo para os dois casos ([E0782] expected a type, found a trait), o compilador apresenta alternativas distintas para a correção de cada ocorrência. Infelizmente, seguindo a recomendação imediata eu não consegui solucionar meu problema. Isso me fez pesquisar mais a fundo o que efetivamente estava acontecendo.

Em resumo: diferente de uma interface em Java, por exemplo, um trait em Rust **não define um tipo**. Isso é relevante porque o compilador precisa saber o tamanho total da nossa struct em tempo de compilação. Como o trait pode ser implementado por structs de diferentes tamanhos, é impossível determinar o tamanho da propriedade 'activation'. Quanto às sugestões do compilador, a primeira inclui o prefixo dyn para notificar o compilador que a propriedade deve ser acessada via [ligação dinâmica](https://pt.wikipedia.org/wiki/Liga%C3%A7%C3%A3o_din%C3%A2mica_(programa%C3%A7%C3%A3o_orientada_a_objetos), essencialmente apagando o tipo da propriedade. Essa alteração soluciona o primeiro erro, porém não resolve nosso problema no construtor.

error[E0277]: the size for values of type `(dyn ActivationFunction + 'static)` cannot be known at compilation time
  --> src\nn_layer.rs:63:10
   |
63 |     ) -> Layer {
   |          ^^^^^ doesn't have a size known at compile-time
   |
   = help: within `Layer`, the trait `Sized` is not implemented for `(dyn ActivationFunction + 'static)` 

Pesquisando mais sobre o problema, encontrei 2 soluções apropriadas para nosso caso:
1. Definir o tipo da propriedade como Box<dyn ActivationFunction>: o tipo Box é utilizado para armazenar ponteiros de objetos alocados dinamicamente. Como o tamanho desse ponteiro é fixo, o compilador nos permite utilizar objetos desse tipo. Nosso construtor seria ajustado para receber o tipo conforme sugestão do compilador:
`activation_function: impl ActivationFunction + 'static ... activation: Box::new(activation_function)`
2. Implementar a struct utilizando um tipo genérico que implementa nosso trait. Essa alternativa é mais próxima do modelo de código que estou acostumado em outras linguagens, permitindo a implementação da estrutura sem nos preocuparmos com a implementação concreta do trait. O mais interessante para mim é que essa foi a primeira sugestão do compilador na mensagem de erro original (help: use a new generic type parameter, constrained by `ActivationFunction`). Para simplificar meu entendimento, preferi seguir com essa versão. O código ajustado é mostrado abaixo:

```
pub struct Layer<T: ActivationFunction> {
    neurons: Matrix,
    weights: Matrix,
    biases: Matrix,
    activation: T, //Verificar como armazenar o objeto de trait
}
impl<T:ActivationFunction> Layer<T> {
   //Cria uma nova camada
   pub fn new(
      prev_layer_neurons: usize,
      layer_neurons: usize,
      activation_function: T, //Verificar se é a melhor forma de armazenar isso aqui
   ) -> Layer<T> {
      Layer {
         neurons: Matrix::new(layer_neurons, 1),   //Matriz coluna
         biases: Matrix::new(layer_neurons, 1), //Matriz coluna
         weights: Matrix::new_random(layer_neurons, prev_layer_neurons),
         activation: activation_function,
      }
   }
```

Essa implementação é uma replicação quase direta da descrição apresentada na minha [fonte original](https://www.3blue1brown.com/lessons/neural-networks#more-compact-notation). A matriz de neurônios consiste de uma matriz com 1 linha por neurônio e apenas 1 coluna. Como cada neurônio tem seu viés, a matriz de viéses (biases) possui a mesma estrutura. A diferença está na estrutura da matriz de pesos: ao invés de representar a conexão com a camada seguinte, temos 1 linha para cada neurônio da camada e 1 coluna para cada conexão vinda da camada anterior. Nesse caso, cada linha **i** da matriz representa os pesos associados ao neurônio **i** dessa camada. Conversamente, cada coluna **j** representa os pesos com origem no neurônio **j** da camada anterior. Essa estrutura garante a validade da multiplicação de matrizes, visto que o número de colunas da matriz será exatamente o número de linhas do vetor de ativação da camada anterior. 

Obs: depois de completar a primeira versão da rede neural descobri uma forma mais elegante de especificar as camadas com uma função de ativação genérica. Descrever no anexo A. 
 

Podemos agora implementar a propagação dos valores na rede neural. Seguindo nossa referência, o valor de ativação de cada neurônio é definido pela soma ponderada do produto dos valores de entrada com os pesos, somado a um valor de viés e utilizado como entrada em uma função de ativação. Matematicamente, temos a definição abaixo:

//a_0 = f((\sum_{0}^{n} v_n * w_{0,n}) + b_0)

Aqui eu cheguei na minha primeira dificuldade com a compreensão da rede neural. Eu entendo a fórmula do somatório, mas como isso é implementado pela multiplicação de matrizes?

A primeira intuição para solucionar minha dúvida está na relação entre o produto escalar (*dot product*) e a multiplicação de matrizes. 
Vamos ignorar o viés e a função de ativação e considerar o caso de um único neurônio da camada: O valor desse neurônio é dado pela multiplicação do valor de entrada (neurônio da camada anterior), pelo peso a ele atribuído: (w * v). Se ele recebe sinais de múltiplos neurônios na camada anterior, seu valor é dado pela soma ponderada de suas entradas (valor multiplicado pelo peso): sum(w_i * v_i).

Se representamos os neurônios da camada de entrada no formato de um vetor e_n, e ordenamos os pesos associados ao neurônio saida (s_n) como um vetor w_n, então o valor do neurônio é dado pelo produto escalar desses vetores. Agrupando o valor resultante de cada neurônio em um novo vetor, temos a representação abaixo:

//https://latex2image.joeraut.com/
s_k =
\begin{bmatrix} w_{00} \\ w_{1k} \\ \vdots \\ w_{nk} \end{bmatrix}
\cdot
\begin{bmatrix} v_{0} \\ v_{1} \\ \vdots \\ v_n  \end{bmatrix} 
=
\sum_{0}^{n} v_n * w_{nk}
\to
S = \begin{bmatrix}
   \sum_{0}^{n} v_n * w_{n0} \\
   \sum_{0}^{n} v_n * w_{n1} \\
   \vdots \\
   \sum_{0}^{n} v_n * w_{nk}
\end{bmatrix}

Continuando nessa intuição, fazemos a transposição dos vetores de peso de cada neurônio e os empilhamos, formando uma matriz onde cada linha representa os pesos associados a 1 neurônio de saída. A partir daqui, se lembramos da regra de multiplicação de matrizes (multiplico a linha da primeira pela coluna da segunda), vemos que o vetor resultante é o mesmo que geramos para o caso de cada neurônio.

\begin{bmatrix} s_{0} \\ s_{1} \\ \hdots \\ s_n  \end{bmatrix}  =
\begin{bmatrix} 
   w_{00} & w_{10} & \hdots & w_{n0} \\
  w_{01} & w_{11} & \hdots & w_{n1} \\
  \vdots & \vdots & \ddots & \vdots \\
  w_{0n} & w_{1n} & \hdots & w_{nn} 
\end{bmatrix}
\begin{bmatrix} v_{0} \\ v_{1} \\ \hdots \\ v_n  \end{bmatrix} 
=
\begin{bmatrix}
   \sum_{0}^{n} v_n * w_{n0} \\
   \sum_{0}^{n} v_n * w_{n0} \\
   \vdots \\
   \sum_{0}^{n} v_n * w_n0
\end{bmatrix} 

Foi assim que eu finalmente entendi a representação matricial dos pesos e valores de ativação da rede neural. 

```
pub fn propagate(&mut self, input_neurons: Matrix) {
   //A ordem importa (weights * input) geraria erro!
   let dot_product = &(self.weights) * &input_neurons;    //Soma ponderada das entradas
   let biased_values = dot_product + &self.biases;        // Soma dos viéses
   assert!(biased_values.rows() == self.neurons.rows());  // Se falhar, a implementação de matriz está errada!
   //Biased_values deve ser uma matriz nx1
   for i in 0..biased_values.rows() {        
      self.neurons[i][0] = self.activation.activate(biased_values[i][0]);
   }
}
```
Neste ponto eu percebi uma ineficiência gritante na implementação. Cada execução do método propagate aloca 2 matrizes novas: a primeira é alocada na multiplicação, enquanto a segunda é alocada para receber o resultado da soma. Por sorte, isso é facilmente solucionado com a implementação de métodos mutáveis para a adição e multiplicação de matrizes com os operadores (+= e *=).

Antes de gerarmos os testes precisamos de ao menos 1 implementação de função de ativação. Buscando facilitar a validação do código, a função de ativação mais simples é a função identidade, que retorna o próprio valor informado. 

```
pub struct Identity {}
impl ActivationFunction for Identity {
    fn activate(&self, val: f64) -> f64 {
        val
    }
    fn derivative(&self, _: f64) -> f64 {
        1.0
    }
}
```

Como podemos gerar uma função de testes se o resultado depende das matrizes de pesos, que são inicialidas aleatóriamente? O conjunto de testes mais simples para o nosso caso é a propagação por 2 camadas.
Como a função de ativação é responsável por introduzir não-linearidade no sistema, uma rede de neural cuja função de ativação é a identidade representa apenas uma transformação linear dos dados de entrada. Essa transformação linear é calculada usando as equações algébricas. Representando o resultado da primeira camada como o produto dos pesos (W) pela entrada (X): A_1 = f(W_1 . X +b_1) -> A_1 = f(W_1 . X +b_1), o resultado da segunda camada da rede neural é dado substituindo essa definição na equação:  

\begin{array}{l}
A_2 = W_2 \cdot A_1 + b2 \\[6pt]
\to W_2 \cdot (W_1 \cdot X + b_1) + b_2 \\[6pt]
\to (W_2 \cdot W_1) \cdot X + (W_2 \cdot b_1 + b_2)\\[6pt]
\end{array}

```
#[cfg(test)]
mod tests {
   // Note this useful idiom: importing names from outer (for mod tests) scope. (Rust Book)
   use super::*;
   #[test]
   fn test_propagate() {
      //Definições do tamanho das camadas
      let input_n = 3;
      let layer1_n = 5;
      let layer2_n = 7;
      let mut layer1 = Layer::new(input_n, layer1_n, Identity {});
      let mut layer2 = Layer::new(layer1_n, layer2_n, Identity {});
      //Mock dos dados da camada de entrada
      let input_mock = Matrix::from_vec(input_n, 1, vec![1.0, 1.0, 1.0]); 
      //Propagação sequencial
      layer1.propagate(&input_mock);
      layer2.propagate(&layer1.neurons);
      //Álgebra para gerar transformação linear equivalente à propagação
      let linear_transform =  &layer2.weights * &layer1.weights; //  W_2 * W_1
      let bias_transform = (&layer2.weights * &layer1.biases) + &layer2.biases; //  W_2 * b_1 + b_2
      let expected = (linear_transform * input_mock) + &bias_transform; // (W_2 * W_1) * X + (W_2 * b_1 + b_2)

      assert!(expected == layer2.neurons);
   }
}
```


Para finalizar, implementamos funções de ativação utilizadas em redes neurais na prática. Escolhi implementar a função sigmoide, usada como exemplo na série de vídeos 3b1b e a função ReLu, descrita como a função mais popular para deep learning no [wikibooks](https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions#ReLU:_Rectified_Linear_Unit). Dada a natureza das funções, a implementação é feita replicando diretamente as [fórmulas](https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions#Continuous_Log-Sigmoid_Function) de cada função na sintaxe Rust:
```
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
```

No próximo artigo vamos finalizar a rede neural com a implementação da retropropagação e do treinamento. Com essa implementação, poderemos finalmente criar e validar uma rede neural com um teste de brinquedo.

Próximos passos nesse código:
1. Ajustar a implementação de propagação para utilizar as versões mutáveis das operações de matrizes;
2. Avaliar alternativas para remover as matrizes de neurônios da definição das camadas, substituindo-as por uma única matriz mutável para representar os dados transmitidos na rede neural.
3. Investigar se a implementação do trait ActivationFunction é a melhor maneira de deixar a função genérica nas camadas;
4. Implementar a propagação para múltiplas entradas de forma paralela, possibilitando o treinamento com várias entradas simultaneamente.


Anexo A: Reimplementação de Layer

Esse exemplo é uma ilustração interessante sobre como uma característica da linguagem força a escrita de código mais robusto e idiomático. Uma dificuldade encontrada na especificação de um módulo exigiu uma refatoração significativa do código para alcançar uma solução adequada. Para um projeto em caráter de aprendizado, essa é uma oportunidade ótima de conhecer a linguagem mais a fundo. Agora, se eu estivesse trabalhando em um sistema complexo, não sei qual seria a minha reação.

Uma breve apresentação do controle de posse de Rust. 
1. Todo valor tem um, e somente um, proprietário;
2. Quando o proprietário sai do escopo, o valor pode ser dealocado;

Quando um valor é atribuído a uma variável, propriedade ou parâmetro de função, essa atribuição pode ocorrer de 3 maneiras:
1. Copy: um valor pode ser copiado, tendo uma nova seção de memória alocada e o seu conteúdo replicado. Nesse caso, o proprietário original detém a propriedade do valor fonte enquanto o valor à esquerda recebe propriedade da cópia. O comportamento de cópia é padrão para tipos inexpensivos (inteiros, floats, chars, etc...).
   let x = Identity {} <-- x é proprietário do objeto 
   processa_fn(x.clone());    <-- Cópia de X
   println!("{}",x);              <-- OK!
2. Move: quando um valor é movido, seu proprietário passa a ser o elemento a quem ele foi atribuído. O dono original é considerado inválido depois que o valor é movido. O comportamento de move é o padrão para structs e tipos complexos.
   let x = Identity {}; <-- x é proprietário do objeto
   processa_fn(x);     <-- parâmetro da função assume a posse do objeto. Quando a função finaliza, a memória de x é dealocada.
   println!("{}",x);       <-- ERRO pois x não é mais uma referência válida! 
3. Borrow: se eu quero fornecer um valor sem gerar uma cópia ou delegar a posse do objeto, podemos fazer um empréstimo. Nesse caso, a posse é transmitida temporariamente":
   let x = Identity {}; <-- x é proprietário do objeto
   processa_fn(&x);     <-- a referência é emprestada para a função. Quando o escopo for finalizado, sua posse volta a variável original (x)
   println!("{}",x);       <-- ERRO pois x não é mais uma referência válida! 

Devido às especificações de controle de posse de um objeto em Rust, implementação original de Layer nos obriga a alocar um objeto ActivationFunction novo a cada camada que especificamos. O problema é ilustrado a seguir:

let id_function = Identity {};
let mut layer1 = Layer::new(input_n, layer1_n, id_function); <-- O objeto referido por id_function foi movido e agora é controlado pela instância layer1
let mut layer1 = Layer::new(input_n, layer1_n, id_function); <-- Erro: a referência id_function não é mais válida!

let mut layer1 = Layer::new(input_n, layer1_n, Identity {}); <-- Novo objeto Identity 
let mut layer1 = Layer::new(input_n, layer1_n, Identity {}); <-- Novo objeto Identity

Embora o custo de memória e performance dessas alocações seja ínfimo, visto que é uma struct vazia, o código não parece limpo o suficiente. Além disso, os módulos que utilizam objetos do tipo Layer são obrigados a considerar o tipo genérico em suas próprias implementações. Isso torna-se um problema quando queremos construir a struct NeuralNetwork para armazenar camadas com funções de ativação distintas.

pub struct NeuralNetwork {
    layers: Vec<Layer<T>>,
    learning_rate: f64,
}

A especificação é inadequada devido ao tratamento de genéricos da linguagem. Essencialmente, o compilador gera uma versão distinta da struct para cada tipo genérico fornecido, transformando-os em tipos concretos. Isso quer dizer que, se usamos a função de ativação Relu o vetor de camadas será do tipo Vec<Layer<Relu>> e não aceitará camadas do tipo Vec<Layer<Sigmoid>>. 

Observando outras implementações da linguagem descobri um mecanismo que simplifica muito nossa implementação. Pensando em termos formais, nossa struct com a função de ativação existe apenas para armazenar 2 ponteiros de função: fn activate e fn derivative. São esses 2 ponteiros que a camada precisa conhecer para executar as chamadas apropriadas, portanto podemos definir as propriedades da struct em termos dessas funções:

pub struct Layer<T: ActivationFunction> {
    neurons: Matrix,
    weights: Matrix,
    biases: Matrix,
    //activation: T, //Substituído pelos ponteiros de função:
    activation_function: fn(f64) -> f64, //Verificar como armazenar o objeto de trait
    activation_derivative: fn(f64) -> f64, //Verificar como armazenar o objeto de trait
}

Com essa estrutura, não temos mais o tipo genérico associado a nenhuma propriedade, portanto podemos remover a propriedade de tipo genérico da struct:

pub struct Layer<T: ActivationFunction> {
    neurons: Matrix,
    weights: Matrix,
    biases: Matrix,
    //activation: T, //Substituído pelos ponteiros de função:
    activation_function: fn(f64) -> f64, //Verificar como armazenar o objeto de trait
    activation_derivative: fn(f64) -> f64, //Verificar como armazenar o objeto de trait
}

Agora precisamos apenas associar as funções definidas para a função de ativação desejada aos ponteiros via construtor. Isso pode ser feito utilizando uma função genérica, com o tipo restringido ao nosso trait ActivationFunction. Nesse caso, o construtor simplesmente associa os ponteiros de função à função apropriada. Como as ActivationFunctions não mantém estado interno, não realizam mutação, portanto são funções puras. Isso 

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

Agora a estrutura é simplificada, sendo necessário apenas informar o tipo da função de ativação na construção dos objetos:

pub struct NeuralNetwork {
    layers: Vec<Layer<T>>,
    learning_rate: f64,
}

fn main() {
   let mut network = NeuralNetwork::new(2, 0.8);
   let input_layer = Layer::new::<Relu>(784, 784);
   let hidden_layer1 = Layer::new::<Relu>(784, 128);
   let hidden_layer2 = Layer::new::<Sigmoid>(128, 128);
   network.add_layer(input_layer);
   network.add_layer(hidden_layer1);
   network.add_layer(hidden_layer2);
}

Com essa definição, todos os objetos Layer tem exatamente o mesmo tipo, porém possuem múltiplos construtores. Um construtor para cada tipo de função de ativação definida. Seria o equivalente a definir 2 funções abaixo. A grande diferença é que, caso criemos uma nova função de ativação, já temos o construtor da camada com aquela função implementado automaticamente.

pub fn new_relu<>(...) -> Layer {
   Layer {
      ...
      activation_function: Relu::activate,
      activation_derivative: Relu::derivative,
   }
}

pub fn new_sigmoid<>(...) -> Layer {
   Layer {
      ...
      activation_function: Sigmoid::activate,
      activation_derivative: Sigmoid::derivative,
   }
}
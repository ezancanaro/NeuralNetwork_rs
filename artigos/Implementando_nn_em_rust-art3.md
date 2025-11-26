## Retropropagação

O autor do vídeo não estava brincando quando disse que leva um tempo para compreender completamente o algoritmo de retropropação. Pessoalmente, a explicação provida no material fonte não foi suficiente para que eu pudesse vizualizar a estrutura do código desse fragmento. Em especial, senti muita dificuldade em aplicar a transição de um único neurônio -> uma função simplificada, para o caso de múltiplos neurônios: operações matriciais.

Tratando de operações de classificação, podemos medir o desempenho de nossa rede neural através de uma função de custo. Essa função de custo nos diz quão próximo o resultado da rede neural está da resposta correta: quanto menor o custo, mais próximo da resposta correta está o resultado. Dessa forma, o desempenho da rede neural pode ser otimizado minimizando o resultado dessa função de custo. Matematicamente, minizamos uma função f(x)=y calculando a direção de ajuste de x (esquerda ou direita) que leva ao menor valor de y. Esse processo pode ser repetido iterativamente, ajustando o tamanho dos passos na direção desejada até alcançar um mínimo local.

Para uma função com múltiplas variáveis, como é o caso da nossa rede neural, com seus pesos e vieses, a direção desses passos é descrita por um vetor, denominado gradiente. O gradiente nos diz em que direção devemos ajustar cada variável, cada peso e cada viés, para que o resultado da rede neural se aproxime do resultado desejado.
Obs: Tecnicamente, o gradiente é um vetor que aponta a direção dos **máximos** da função, portanto, o ajuste é feito através da subtração deste gradiente de nossos pesos e vieses.   

Como o número de pesos e vieses em uma rede neural extende-se além dos milhares, calcular os valores desse vetor diretamente é uma tarefa hercúlea. O algoritmo de retropropagação oferece uma alternativa para que esse cálculo seja realizado iterativamente. Partindo da camada de saída da nossa rede neural, o gradiente de cada camada pode ser calculado com base nos valores de sua vizinhança.  

## A Camada de Saída

Embora eu não tenha compreendido completamente a explicação do 3b1b, uma coisa ficou clara: o cálculo da retropropagação da camada de saída da rede neural é distinto das camadas ocultas. Quando consideramos que o cálculo dos gradientes depende das camadas vizinhas, isso faz muito sentido intuitivamente, já que a camada final possui apenas a camada anterior como vizinho imediato. O gradiente dessa camada é calculado com base em 3 fatores:
1. Valor de ativação da camada anterior;
2. Pesos da camada de saída;
3. Vieses da camada de saída;

Novamente temos uma sensação intuitiva aqui: esses 3 fatores são responsáveis pelo valor de saída da camada. Portanto, se eu preciso modificar o valor de saída para minimizar o valor de custo, esses 3 fatores devem ser ajustados.
Relembrando os conceitos de cálculo, a "rampa" (taxa de variação) de uma função em relação a uma variável é dada pela derivada da função em relação àquela variável. Para funções multivariadas, a taxa de variação em relação a cada variável é dada por derivadas parciais. Com essa memória ativada, podemos revisar o cálculo do gradiente da camada final.

O custo da rede neural (C_l) é dado pelo quadrado da diferença entre ativação da camada de saída (a_l) e o resultado esperado: C_l = (a_l - y)^2. Para minizar esse custo, precisamos ajustar os parâmetros da função (a ou y). O resultado esperado é fixo, portanto precisamos ajustar o valor de a_l. Queremos saber quanto o custo da função é alterado por modificações no valor de a, portanto temos a derivada parcial ∂C/∂a

Como esse parâmetro advém de uma rede neural, não podemos modificar diretamente os valores de a_l, portanto, precisamos ajustar os parâmetros que compõe a_l : (a_l = σ(w . a_l-1 + b)). 
Não modificaremos a função de ativação, portanto o valor de a_l é afetado pelas constituintes da soma ponderada usada como input dessa função: z = w . a_l-1 + b. A atribuição da variável z a essa soma ponderada nos permite uma representação simplificada da derivada parcial ∂a/∂z, representando o quanto uma alteração no valor da soma afeta o valor de a.

Persistindo na decomposição dos termos, precisamos saber o quanto cada componente da soma ponderada afeta o seu valor. No cálculo de z, podemos manipular o valor dos pesos w e dos vieses b. Portanto, a taxa de variação de z em relação aos pesos é dada por ∂z\∂w e a taxa de variação em relação aos vieses é dada por ∂z\∂b.

Lembrando que o objetivo objetivo final é obter o gradiente da função de custo em relação aos pesos e vieses da camada, pois são os únicos parâmetros que podemos alterar diretamente. Portanto, precisamos obter ∂C\∂w: a derivada parcial de C em relação aos pesos; e ∂C\∂b derivada parcial de C em relação aos vieses. Essas derivadas parciais são obtidas através da aplicação da [regra da cadeia](https://pt.wikipedia.org/wiki/Regra_da_cadeia), agrupando as variáveis de cada componente da saída, apresentadas acima:

∂C\∂w = ∂z\∂w . ∂a/∂z . ∂C/∂a
∂C\∂b = ∂z\∂b . ∂a/∂z . ∂C/∂a

Embora a definição dessas variadas já seja dada pela fonte principal, as equações não são complexas de diferenciar "na mão".

∂z\∂w -> z = w . a_l-1 + b. Para uma derivada parcial, consideramos apenas a variável relacionada, tratando as demais como uma constante qualquer. Aplicamos 2 regras básicas de derivadas: A derivada de uma constante é sempre 0 e a derivada de um exponencial é dada pela regra da potência (d/dx (x^n) = nx^n-1).
Portanto, temos a derivação ∂z\∂w = w . a_l-1 + b -> 1.a_l-1 + 0 -> a_l-1

∂a/∂z -> a = σ(z). É simplesmente a derivada da função de ativação aplicada ao valor de entrada **z**. O valor dessa derivada depende da função de ativação utilizada σ'(z). 

∂C/∂a -> C = (a - y)^2 = 2(a - y).  O detalhe que eu esqueci quando tentei chegar nesse resultado pela primeira vez, e que pega muito estudante de cálculo de calça curta, é que devemos considerar (a - y) como uma função (f): C = f^2. Portanto, a derivada deve ser obtida pela regra da cadeia: ∂C/∂a = ∂C/∂f . ∂f/∂a.
Assim temos: ∂C/∂f = 2f (pela regra da potência) e ∂f/∂a = 1 pelas regras da potência (a) e da constante (y). Substituindo na regra da cadeia:  
∂C/∂a = ∂C/∂f . ∂f/∂a = 2f . 1 = 2(a - y)

∂z\∂b -> z = w . a_l-1 + b. Na derivada parcial, tratamos w e a_l-1 como constantes, portanto o produto w . a_l-1 tem derivada 0. A derivada do termo b é otido pela regra da potência: d\db(b) = 1b^0 = 1

Assim sendo, temos todos os componentes matemáticos para implementar a retropropagação. Finalmente podemos voltar a ver código aqui. Para começar, o cálculo depende do valor da soma ponderada z, que é calculado durante a propagação dos valores na rede neural, implementada no artigo anterior. Não faz muito sentido recalcular esse valor, portanto modificamos o código das camadas para armazenar o valor dessa soma durante a propagação:

```
pub struct Layer {
    neurons: Matrix,
    zed: Matrix,            //Variável z na retropropagação
    weights: Matrix,
    biases: Matrix,
    activation_function: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64, 
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
            self.zed[i][0] = biased_values[i][0]; //armazena o valor de z para a retropropagação
            self.neurons[i][0] = (self.activation_function)(biased_values[i][0]);
        }
    }
```

Um detalhe importante é que a derivada parcial deve ser tomada em relação a cada um dos pesos e vieses da camada de saída. Portanto, podemos visualizar essas derivadas em formato de matrizes (Essa visualização secundária da explicação foi essencial para que eu entendesse por completo: https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0/):

∂C\∂w = ∂z\∂w . ∂a/∂z . ∂C/∂a

|∂z\∂w  ∂z\∂w  ... ∂z\∂w|     |∂a\∂z|     |∂C\∂a|     
|∂z\∂w  ∂z\∂w  ... ∂z\∂w|     |∂a\∂z|     |∂C\∂a|    
|∂z\∂w  ∂z\∂w  ... ∂z\∂w|     ...         |∂C\∂a|        
|∂z\∂w  ∂z\∂w  ... ∂z\∂w|     |∂a\∂z|     |∂C\∂a|   

Depois disso podemos iniciar a implementação da retropropagação para a camada de saída da rede neural. Para esse cálculo, nossa função deve receber 3 parâmetros: 
 1. O valor de saída esperado para nossa rede;
 2. O valor de ativação da camada anterior;
 3. A derivada da função de custo
Ela retornará o gradiente dessa camada, representado por uma única matriz. Implementamos a função com uma tradução direta das derivadas aplicadas a cada neurônio da nossa rede neural.
2 detalhes são dignos de nota:
 1. O cálculo da derivada parcial em relação aos viéses implícito na função, sendo esse gradiente armazenado na matriz deltas;
 2. O gradiente dos pesos pode ser representado como uma matriz, que armazena o valor de ajuste de cada camada. Essa intuição não estava clara na explicação inicial, que trata o gradiente como um vetor único para **todos** os parâmetros da rede. 

```
pub struct Layer {
    neurons: Matrix,
    ...,
    weight_derivatives: Matrix, // Gradiente de erro dos pesos da camada
}

pub fn cost_derivative(activation_val: f64, expected_val: f64) -> f64 {
        2 * (activation_val - expected_val)
}

pub fn backpropagate_output_layer(
        &mut self,
        expected: &Matrix,
        prev_activations: &Matrix,
        cost_derivative: impl Fn(f64, f64) -> f64,
    ) {
        let mut deltas = Matrix::new(self.neurons.rows(), 1)
        for i in 0..self.neurons.rows() {
            //∂C/∂a = 2(a - y) - Derivada parcial de C por a
            let c_a_partial_derivative = cost_derivative(self.neurons[i][0], expected[i][0]);
            //∂aL/∂z = activation'(z) - Derivada parcial de a por z
            let a_zed_partial_derivative = self.activation.derivative(self.zed[i][0]);
            //δ = hadamard_product(∂C/∂a, ∂aL/∂z).
            //Detalhe: o vetor delta é a derivada em função dos viéses ∂C/∂b = ∂z/∂b * ∂a/∂z * ∂C/∂a já que ∂z/∂b = 1
            deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;
            for j in 0..self.weights.cols() { //Para cada peso, calcula a derivada parcial em relação ao valor desse neurônio
                //∂C/∂w
                //∂z/∂w = a_(L-1).
                self.weight_derivatives[i][j] = prev_activations[j][0] * deltas[i][0];
            }
        }
    }
```

## Camadas Ocultas 

Na camada de saída, determinamos a taxa de variação da função de custo com base na ativação final (a) da nossa rede neural. Precisamos agora repetir o processo para as camadas ocultas, determinando a taxa de variação da função de custo em relação a ativação de cada camada. Relembrando, nossa rede neural é composta por N camadas, ordenadas de 0 a N. Essas camadas podem ser representadas pelo conjunto {0, 1, ..., N-2, N-1, N}. 

////REESCREVER ESSA SEÇÃO
*A taxa de variação da função de custo em relação à camada final (∂C/∂a_n) já foi calculada previamente. Como a retropropagação funciona em passos "para trás", o próximo gradiente a ser calculado refere-se à camada anterior: ∂C/∂a_n-1. A grande diferença para esse cálculo em relação ao gradiente da camada de saída está na conexão dos neurônios dessa camada. Para a camada final, o valor de ativação dos neurônios forma um vetor que tem uma relação direta com o vetor do resultado esperado. Se imaginamos o resultado esperado como um conjunto de neurônios, podemos dizer que cada neurônio da camada de saída está diretamente conectado com apenas 1 neurônio do resultado: seu par na mesma posição. Portanto, a alteração do valor de um neurônio da camada de saída impacta apenas 1 neurônio do resultado esperado.

Para as camadas ocultas, isso não é verdade. Na nossa rede neural densa, cada neurônio da camada n-1 está conectado com **todos** os neurônios da camada **n**. Isso quer dizer que alterações no valor de um neurônio na camada n-1 impactam o valor de **todos** os neurônios da camada **n**. Para determinar como o valor da ativação **a_N-1** impacta o valor da função de custo, precisamos traçar todas as conexões dessa camada com a próxima. Retomando a definição da ativação a_N = σ(w . a_N-1 + b), temos que o efeito da ativação da camada N-1 se manisfeta apenas na soma ponderada, nosso z_N. Portanto, a taxa de variação da função de custo em relaçaõ a ativação **a_n-1** é dado em termos da taxa de variação de **z_N** em relação à ativação **a_N-1**: ∂z/∂a_n-1.*

Como o valor final da rede neural (a_N) continua sendo afetado pelos parâmetros da camada de saída, os demais termos da regra da cadeia permanecem na equação:
∂C/∂a_N-1 = ∂z/∂a_n-1 . ∂a/∂z_N . ∂C/∂a_N

As derivadas parciais para os termos ∂a/∂z_N e ∂C/∂a_N continuam exatamente as mesmas. Precisamos obter apenas o valor da derivada ∂z/∂a_n-1 considerando a definição de z. A lógica para a diferenciação é exatamente a mesma usada em LINK, com a única alteração existindo no termo que estamos diferenciando (a_N-1 ao invés de w):
∂z/∂a_n-1(w . a_l-1 + b) = w. 1.a_N-1^0 + 0 = w  

Novamente, não podemos alterar diretamente o valor da ativação da camada (n-1), portanto precisamos "quebrar" a derivada ∂z/∂a_n-1 para tratá-la em relação aos parâmetros ajustáveis: pesos e vieses. Usamos os mesmos passos descritos para a camada de saída e obtemos as derivadas parciais ∂a_N-1/∂z_N-1, ∂z_N-1\∂w_N-1 e ∂z_N-1\∂b_N-1. Substituindo na equação original, temos as derivadas parciais em relação aos pesos e vieses da penúltima camada:

∂C\∂w_N-1 = ∂z_N-1\∂w_N-1 . ∂a_N-1/∂z_N-1 . ∂z^j_N /∂a_n-1  . ∂a/∂z_N . ∂C/∂a_N
∂C\∂b_N-1 = ∂z_N-1\∂b_N-1 . ∂a_N-1/∂z_N-1 . ∂z^j_N /∂a_n-1  . ∂a/∂z_N . ∂C/∂a_N

O primeiro detalhe dessa equação: o produto ∂a/∂z_N . ∂C/∂a_N é o mesmo obtido no cálculo da retropropação para a camada de saída. Uma repetição similar será encontrada se expandirmos a equação para a camada N-2: o produto ∂a_N-1/∂z_N-1 . ∂z^j_N /∂a_n-1 será repetido. Essas repetições ocorrem sucessivamente a cada passo de propagação. Para não recalcular esses valores toda vez, adicionamos uma matriz `delta` como membro da struct que representa nossas camadas. Também ajustamos a função de retropropagação da camada de saída para armazenar esses valores:

Obs: o nome delta é usado porque o símbolo (𝜹) é comumente utilizado para representar esse termo nas equações de retropropagação demonstradas nos materiais de referência. 

```
pub struct Layer {
    neurons: Matrix,
    zed: Matrix,            //Variável z na retropropagação
    deltas: Matrix,         //Vetor de erro / gradiente de vieses.
    weight_derivatives: Matrix, // Gradiente de erro dos pesos da camada 
    weights: Matrix,
    biases: Matrix,
    activation_function: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64, 
}
pub fn backpropagate_output_layer(...) {
    //let mut deltas = Matrix::new(self.neurons.rows(), 1) Removemos a variável temporária
    for i in 0..self.neurons.rows() {
        ///...
        self.deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;
        for j in 0..self.weights.cols() { 
            self.weight_derivatives[i][j] = prev_activations[j][0] * self.deltas[i][0];
        }
    }
}
```

É essa repetição de termos que justifica a lógica da retropropagação: os termos são calculados uma única vez e propagados para trás.


Para as camadas ocultas, isso não é verdade. Na nossa rede neural densa, cada neurônio da camada n-1 está conectado com **todos** os neurônios da camada **n**. Isso quer dizer que alterações no valor de um neurônio na camada n-1 impactam o valor de **todos** os neurônios da camada **n**. Portanto, o formato real da derivada  ∂C/∂a_N-1 = ∂z/∂a_n-1 . ∂a/∂z_N . ∂C/∂a_N é o somatório de todas as conexões dessa camada: 

∂C/∂a_N-1 = ∑_j=0 ∂z^j_N /∂a_n-1 . ∂a^j_N/∂z^j_N . ∂C/∂a^j_N 


Com essa observação, a função de retropropagação para as camadas ocultas depende de 2 fatores:
1. Os valores do produto ∂a/∂z_N . ∂C/∂a_N e dos pesos da camada seguinte. Em nosso código, passamos uma referência à camada completa (&next_layer);
2. Os valores de ativação da camada anterior, transmitidos diretamente em formato de matriz.

```
pub fn backpropagate_hidden_layer(
        &mut self,
        next_layer: &Layer<T>,
        prev_activations: &Matrix,
    ) {
    //Transposição para que as dimensões estejam compatíveis.
    //Desnecessária pois wt[i][j] == w[j][i]
    //let weight_transpose = next_layer.weights.transpose();
    for i in 0..self.neurons.rows() {
        //∂aL/∂z = activation'(z) - Derivada parcial de a por z
        let mut c_a_partial_derivative = 0.0;
        //∂z/∂a_(l-1) * δ_l
        for j in 0..next_layer.weights.rows() {
            c_a_partial_derivative += next_layer.weights[j][i] * next_layer.deltas[j][0];
        }
        let a_zed_partial_derivative = self.activation.derivative(self.zed[i][0]);
        //δ = ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
        self.deltas[i][0] = c_a_partial_derivative * a_zed_partial_derivative;
        for j in 0..self.weights.cols() {
            //∂z/∂w = a_(L-1).
            //∂C/∂Cw_(l-1) = ∂z_(l-1)/∂w_L-1 * ∂a_(l-1)/∂z_(l-1) * sum(∂z_l/∂a_(l-1) * δl)
            //∂C/∂Cw_(l-1) = a_(L-1) * δ
            self.weight_derivatives[i][j] = prev_activations[j][0] * self.deltas[i][0];
        }
    }
}
```

Detalhe importante: a matriz de pesos da camada seguinte deve ser transposta na implementação direta. Esse passo não havia ficado claro para mim nas exposições que eu utilizei como base, porém uma análise da estrutura da rede neural deixa o motivo bem evidente.

Lembrando que as dimensões da matriz de pesos da camada L são dadas por **neurons_L** X **neurons_L-1**. Isso significa que o número de neurônios dessa camada é igual ao número de colunas da matriz de pesos da camada seguinte. Nosso laço de repetição é criado com base no número de neurônios da camada atual (i in 0..self.neurons.rows()), portanto, se precisamos processar a matriz de pesos linha a linha, precisamos da transposta para que o número de linhas respeite o pressuposto da multiplicação de matrizes (M x N -> N x P).

A grande questão é que a operação de transposição é desnecessária. Podemos simplesmente acessar a matriz invertendo os índices, de forma que i represente a coluna e j represente a linha. Essa equivalência é [sutilmente apontada](https://www.3blue1brown.com/lessons/backpropagation-calculus#calculating-the-gradient-with-backpropagation:~:text=Those%20indices%2C,backwards%20at%20first%2C) pelo material do 3b1b na inversão dos índices na representação da matriz de pesos.

"Those indices, jk, might feel backwards at first, but it lines up with how you’d index the weight matrix"

//Calculo do teste da retropropagação nas camadas ocultas:
//https://matrixcalc.org/#transpose%28%7B%7B1%2e1,1%2e2,1%2e3,1%2e4%7D,%7B1%2e5,1%2e6,1%2e7,1%2e8%7D,%7B1%2e9,2%2e0,2%2e1,2%2e2%7D%7D%29*%7B%7B0%2e9%7D,%7B-0%2e5%7D,%7B0%2e2%7D%7D

Um ponto chave do algoritmo de retropropagação é a equivalência do vetor **deltas** com o gradiente de custo em relação aos viéses da camada. É fácil de ignorar essa correspondência na explicação apresentada no material base pois a variável **delta** nunca é definida no processo. Novamente, a simplificação auxilia na compreensão intuitiva mas não é a melhor fonte para uma implementação desse algoritmo.
A intuição chave é apresentada na seção abaixo>:

Luckily, this new derivative is simply 1: //ilustração da equação
So the derivative for the bias turns out to be even simpler than the derivative for the weight.
//Ilustrar na equação como as derivadas parciais são exatamente o vetor delta calculado no algoritmo, multiplicados pelo elemento neutro 1.

Outra intuição desnecessária é a conversão dos termos em um vetor único. Como cada camada só é ajustada com o gradiente de seus pesos e viéses, não há necessidade de propagar o vetor completo pela rede. Apenas a camada imediatamente seguinte é relevante para os cálculos dos gradientes.

A escrita dos testes da retropropagação segue a mesma lógica dos testes de propagação: fixamos os valores das camadas de nossa rede neural e utilizamos uma calculadora de matrizes para gerar o resultado esperado. O código foi anexado no final do artigo.

## Treinando a Rede Neural (E minha paciência)

Com todas as peças criadas, agora nos resta "encaixar os bloquinhos" para que a rede neural faça alguma operação útil. Nosso primeiro passo é especificar uma `struct` que representa nossa rede completa:

```
struct NeuralNetwork {
    layers: Vec<Layer<Relu>>
}
```

Criamos a função de treinamento passo a passo. A função deverá receber como parâmetro uma matriz representando a entrada e uma segunda matriz com o resultado esperado.
Primeiro,a função deve classificar os dados de entrada e gerar sua saída. Iniciamos propagando a representação matricial da entrada na primeira camada de nossa rede neural para gerar a primeira representação. Depois precisamos apenas propagar essa representação camada por camada, percorrendo a lista completa sequencialmente:

```
pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
    assert!(!self.layers.is_empty());
    let last_layer_index = self.layers.len() - 1;
    //Propaga a primeira camada
    self.layers[0].propagate(&input);
    //Propaga as camadas remanescentes
    for i in 1..self.layers.len() {
        self.layers[i].propagate(self.layers[i-1].neurons());
    }
    ...
}

```

Esse fragmento de código já dá uma dica do problema que enfrentei na frente. No último artigo apresentei parte das regras de propriedade da linguagem que formam um controle robusto de memória. Outro detalhe dessas regras mostra a cara nesse ponto: 
1. Um objeto pode conter **n** referências imutáveis emprestadas. 
2. Apenas 1 referência mutável pode ser emprestada em um dado escopo.
3. Se houve um empréstimo mutável, não é possível emprestar uma referência imutável no mesmo escopo. O mesmo é válido na direção oposta.

Quem já trabalhou com programação distribuída entende om quanto essas regras impactam para evitar condições de corrida e leituras inválidas. Essencialmente, as regras garantem que múltiplos usuários podem ler o conteúdo de uma seção de memória simultaneamente, desde que nenhum delas queira escrever nessa seção. Assim que alguém declara intenção de escrita (empréstimo mutável), a linguagem impede que outros usuários leiam aquele endereço de memória, ou declarem intenção de escrita, até que as operações de escrita (o escopo do empréstimo) sejam finalizadas. Isso garante que não haverão tentativas simultâneas de escrita e que os leitores não acessarão memória inválida (em processo de escrita).

Embora as regras sejam bem vindas no contexto de programação paralela, este conjunto faz com que a linha `self.layers[i].propagate(self.layers[i-1].neurons());` seja inválida. Para evidenciar esse ponto precisamos da definição da função propagate: `pub fn propagate(&mut self, input_neurons: &Matrix)`.

Seguindo a definição, o parâmetro `self` (o objeto no qual o método é chamado) deve ser uma referência mutável. Isso é auto evidente, já que a propagação deverá alterar a camada. O ponto de conflito aquie é que o acesso aos elementos de um array de camadas da rede neural é feito através de empréstimo implícito do array, utilizando o tipo exigido pelo elemento no contexto. Isso significa que temos uma referência mutável ao array `self.layers[i]` para obter a camada atual. Em contraste com essa referência, a matriz de neurônios da camada anterior é recebida como uma referência imutável. Como a camada anterior está armazenada no mesmo vetor, temos uma referência imutável ao array no acesso `self.layers[i-1].neurons(),` portanto violamos a regra 3 e o compilador nos impede de fazer m*. 

Para que a operação tenha sucesso precisamos de 2 referências distintas aos elementos. Essas referências podem ser obtidas partindo o array em 2 fatias: a primeira contém as camadas que já foram propagadas, enquanto a segunda apresenta as camadas que ainda devem processar os dados. Como uma das referências deve ser mutável, precisamos utilizar o método `split_at_mut`. Utilizando o índice da camada atual como parâmetro dessa função temos exatamente as janelas desejadas, resultando na implementação abaixo:

```
pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
    assert!(!self.layers.is_empty());
    let last_layer_index = self.layers.len() - 1;
    //Propaga a primeira camada
    self.layers[0].propagate(&input);
    //Propaga as camadas remanescentes
    for i in 1..self.layers.len() {
        //Separa em 2 slices: [0..i) e [i..len)
        //Necessário para lidar com o borrow checker de Rust
        let (prev_layers, layers_to_propagate) = self.layers.split_at_mut(i);
        layers_to_propagate[0].propagate(prev_layers[i - 1].neurons());
    }
    ...
}

```

Agora temos 1 referência mutável ao *slice* `layers_to_propagate`, cujo primeiro elemento é a camada que será propagada no momento, e uma referência imutável a `prev_layers`, que inclui todas as camadas já processadas. Nesse cenário, o último elemento de `prev_layers` é a camada processada anteriormente, portanto i-1.

Com a classificação da entrada feita pela propagação, agora é necessário calcular e retropropagar o erro para permitir a rede aprenda algo com essa operação. Já implementamos métodos distintos para a retropropagação na camada de saída e nas camadas ocultas. Começamos então pela camada de saída da rede neural:

```
//Derivada da função de custo
pub fn cost_derivative_mse(x: f64, y: f64) -> f64 {
        2.0 * (x - y)
} 

pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
    //propaga o inpput em toda a rede neural
    ...
    //Limita o escopo dos slices para evitar erro de borrow na retropropagação
    {
        let (hidden_layers, output_layers) = self.layers.split_at_mut(last_layer_index);
        output_layers[0].backpropagate_output_layer(
            &expected_output,
            hidden_layers[last_layer_index - 1].neurons(),
            NeuralNetwork::cost_derivative_mse,
        );
    }
    ...
}
```

Usamos a mesma técnica de fatiar o array de camadas para evitar os erros devido ao empréstimo mutável da camada de saída. Para evitar que esse problema seja constante nos próximos passos, utilizei um bloco de código para criar um escopo limitado e reduzir o tempo de vida dos *slices* criados para essa etapa. Isso garante que não existirão referências ao array de camadas após essa seção de código. Pensando lógicamente, extrair essa seção de código para uma função auxiliar seria uma solução mais "bonita".

Com a camada de saída resolvida, percorremos as camadas ocultas de trás para a frente, executando a retropropagação até a 3ª camada:

```
pub fn train(&mut self, input: Matrix, expected_output: Matrix) {
    // propaga o input em toda a rede neural
    ...
    for i in (2..last_layer_index).rev() { //Inverte o range para percorrer de N até 2
        //slices [0..i) e [i..len()] (Novamente lidando com borrow checker)
        let (propagation_layers, done_layers) = self.layers.split_at_mut(i);
        let (coming_layers, current_layers) = propagation_layers.split_at_mut(i - 1);
        current_layers[0]
            .backpropagate_hidden_layer(&done_layers[0], coming_layers[i - 2].neurons());
    }
    let (remaining_layers, done_layers) = self.layers.split_at_mut(2);
    ...//tratar segunda camada da rede
}
```
Nesse ponto a tática de criar múltiplos *slices* para o array de camadas começa a demonstrar sua fragilidade. Para a retropropagação das camadas ocultas, é necessário obter 3 referências distintas ao array de camadas:
1. A camada atual: i-1;
2. A camada posterior: i, de onde precisamos obter os pesos e o vetor delta;
3. A camada anterior: i-2, contendo os valores de ativação.

Isso torna o código de divisão do array dentro do laço de repetição bem confuso, sendo necessário manipular 4 slices distintos e equilibar os índices de cada um deles em relação ao índice da camada inicial. Também precisamos tratar o caso da 2ª camada separadamente, do contrário o índice [i-2] resultará em uma posição obviamente inválida (1-2 = -1).

Foi aqui que eu decidi dar um passo atrás e repensar a definição dessas funções antes de brigar novamente com o compilador.

## Reescrita do código para resolver problemas do borrow-checker

Um primeiro ponto que pode melhorar a clareza do código é desvincular os gradientes gerados pela retropropagação da própria camada. Isso é muito relevante pois, em uma aplicação real, o valor final de ajuste dos parâmetros da camada não é dado pelos gradientes de um único caso de treinamento, mas sim da média dos gradientes gerados para cada caso apresentado à nossa rede. 
Para que isso seja possível, a função de retropropação deve retornar os gradientes calculados para os pesos e para os vieses da camada. Agrupamos os dois em uma struct de dados e ajustamos nossa camada e as funções de retropropagação apropriadamente:

```
pub struct Gradient {
    pub weight: Matrix,
    pub delta: Matrix,
}
//Removemos delta e weight_derivatives da struct
pub struct Layer {
    neurons: Matrix,
    zed: Matrix,
    weights: Matrix,
    biases: Matrix,
    activation_function: fn(f64) -> f64,
    activation_derivative: fn(f64) -> f64, 
}

pub fn backpropagate_output_layer(
    &mut self,
    expected: &Matrix,
    prev_activations: &Matrix,
    cost_derivative: &dyn Fn(f64, f64) -> f64,
) -> Gradient {
    let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
    let mut deltas = Matrix::new(self.neuron_qty(), 1);
    ... //Cálculo permanece o mesmo
    //Retorna um objeto Gradient transportando as matrizes
    Gradient {
        weight: weight_derivatives,
        delta: deltas,
    }
}

pub fn backpropagate_hidden_layer(
    &mut self,
    next_layer_weights: &Matrix, //Separa pesos do gradiente
    next_layer_deltas: &Matrix, 
    prev_activations: &Matrix,
) -> Gradient {
    let mut weight_derivatives = Matrix::new(self.weights.rows(), self.weights.cols());
    let mut deltas = Matrix::new(self.neuron_qty(), 1);
    ... //Cálculo permanece o mesmo
    Gradient {
        weight: weight_derivatives,
        delta: deltas,
    }
}
```

```
let mut gradients: VecDeque<Gradient> = VecDeque::with_capacity(self.layers.len());
{
    let (hidden_layers, output_layers) = self.layers.split_at_mut(last_layer_index);
    let gradient = output_layers[0].backpropagate_output_layer(
        &expected_output,
        hidden_layers[last_layer_index - 1].neurons(),
        &NeuralNetwork::cost_derivative_mse,
    );
    gradients.push_front(gradient);
}
```

Nossa função de treinamento é ajustada para armazenar esses gradientes em uma coleção temporária. Também consegui simplificar a gestão dos índices das camadas no processamento das camadas ocultas, cortando o array em torno da camada atual de processamento:
1. Um slice contendo as camadas que serão processadas posteriormente é gerado quando dividimos no índice da camada atual **i**. A camada i-1 é sempre o último elemento desse vetor, portanto [i-1] é um acesso válido enquanto nosso range não passa de 1;
2. A camada atual é isolada das demais pela criação de um *slice* com um único elemento. Nessa operação, a camada i+1 é o primeiro elemento do segundo *slice*. O acesso ao índice [0] é sempre válido se o range inicia na penúltima camada do array;
3. Por fim, simplificamos o acesso ao gradiente de erros da camada posterior com a ordem de adição dos elementos nesse vetor. Se adicionamos sempre no início da coleção, o último gradiente calculado será sempre obtido pelo primeiro elemento do vetor. Essa lógica de inserção é o motivo de utilizarmos VecDeque em lugar de Vec para armazenar os gradientes.

```
for i in (1..last_layer_index).rev() {
    //slices [0..i) e [i..len()-1) (Novamente lidando com borrow checker)
    let (initial_layers, current_and_done_layers) = self.layers.split_at_mut(i);
    let (current_layer, done_layers) = current_and_done_layers.split_at_mut(1);
    let gradient = current_layer[0].backpropagate_hidden_layer(
        &done_layers[0].weights(),
        &gradients[0].delta,
        initial_layers[i - 1].neurons(),
    );
    gradients.push_front(gradient);
}
```

Não é necessário executar a retropropagação da camada de entrada, visto que seu valor depende somente da representação da nossa entrada.

Com a retropropagação implementada, o aprendizado da rede neural é concluído com o ajuste dos parâmetros de cada camada:

```
assert!(gradients.len() == self.layers.len() - 1);
//Pulo 1 elemento pois não devo ajustar a camada de entrada
let adjustable_layers = self.layers.iter_mut().skip(1); 
//zip: agrupa 2 iteradores. O laço é finalizado quanto um deles chega ao fim.
//No nosso caso, ambos terão o mesmo tamanho, dado o assert! acima.
for (layer, gradient) in adjustable_layers.zip(gradients) {
    layer.adjust_parameters(gradient, self.learning_rate);
}
```

## Uma tangente sobre o uso de I.A. GEMINI me enganando

Comentário sobre o teste de backprop: O Gemini me fez perder tempo revalidando as operações devido a
uma premissa errônea por parte da IA. 


Avaliação da Lógica do Teste
Estrutura do Teste: Correta 👍
A estrutura do seu teste é excelente. Você seguiu a "receita" para um teste previsível:
....
Cálculo do Resultado Esperado: Incorreto 👎
O problema está no cálculo manual da matriz expected_derivatives. A sua ilustração parece não considerar o efeito da derivada da função de ativação ReLU.
2. Cálculo do Erro delta
A fórmula do delta para a camada de saída é δ=(A−Y)⊙ReLU 
′
 (Z), onde Y é o valor esperado.


Verificando a análise eu fiquei surpreso com a resposta, visto que os cálculos apresentados pela ferramenta não pareciam corretos. Mesmo assim, eu refiz todas as operações na calculadora de matrizes e cheguei na conclusão de que eu não estava errado. 
//https://matrixcalc.org/#2*%28%7B%7B0%2e1,0%2e2,0%2e3,0%2e4,0%2e5%7D,%7B0%2e6,0%2e7,0%2e8,0%2e9,1%2e0%7D,%7B1%2e1,1%2e2,1%2e3,1%2e4,1%2e5%7D%7D*%7B%7B0%2e5%7D,%7B0%2e5%7D,%7B0%2e5%7D,%7B0%2e5%7D,%7B0%2e5%7D%7D+%7B%7B0%2e01%7D,%7B0%2e02%7D,%7B0%2e03%7D%7D-%7B%7B1%2e0%7D,%7B1%2e0%7D,%7B1%2e0%7D%7D%29

Observando novamente a saída do Gemini, percebi um detalhe incidioso:

"A fórmula do delta para a camada de saída é δ=(A−Y)⊙ReLU´(Z), onde Y é o valor esperado."

Não, a fórmula usada não é essa. Fiz os apontamentos para o colega:

'Sobre a seção "ua ilustração [ -0,48 | 2,04 | 4,56] parece ter usado a derivada da função de custo do Erro Quadrático Médio (2 * (A - Y)) mas não a multiplicou pela derivada da ReLU, ou assumiu que a derivada da ReLU era 2.0 para a primeira linha, o que é incorreto. O valor delta correto para a primeira linha é -0.24.":
Seu cálculo do delta está usando (A-Y) como a derivada da função de custo na fórmula δ=(A−Y)⊙ReLU′(Z).
A referência usa 2*(A-Y). Como o valor resultante da derivada da ReLU é um vetor com todos os valores 1.0, a multiplicação deste vetor por outro vetor com o mesmo número de elementos não tem efeito.
Revalide sua análise com base nessas pontuações'










///OUTPUT COMPLETO DO GEMINI:
Com base na sua descrição, o teste parece estar no caminho certo ao fixar todos os valores para garantir um resultado previsível, mas há um erro de cálculo no resultado esperado. A lógica para o delta da primeira linha parece incorreta devido à natureza da função de ativação ReLU.

Avaliação da Lógica do Teste
Estrutura do Teste: Correta 👍
A estrutura do seu teste é excelente. Você seguiu a "receita" para um teste previsível:

Fixou os Parâmetros: A matriz de pesos (weights_mock) e o vetor de viés (bias_mock) não são aleatórios, mas sim fixos com valores conhecidos.

Fixou as Entradas: Os vetores expected_mock e previous_mock também são fixos.

Simulou o Forward Pass: A chamada output_layer.propagate(&previous_mock) calcula e armazena corretamente os estados internos (zed e neurons) necessários para a retropropagação.

Comparou com um Resultado Esperado: O passo final assert!(weight_derivatives == expected_derivatives) é o objetivo de todo teste unitário.

Cálculo do Resultado Esperado: Incorreto 👎
O problema está no cálculo manual da matriz expected_derivatives. A sua ilustração parece não considerar o efeito da derivada da função de ativação ReLU.

Vamos recalcular o delta passo a passo:

1. Cálculo da Passagem Para Frente (zed e neurons)
Entrada (X): Vetor 5x1 com [0.5] em todas as posições.

Pesos (W): Sua matriz 3x5 [0.1, 0.2, ..., 1.5].

Viés (b): [0.01, 0.02, 0.03].

zed = W ⋅ X + b

Linha 1: (0.1 + 0.2 + 0.3 + 0.4 + 0.5) * 0.5 + 0.01 = 1.5 * 0.5 + 0.01 = 0.75 + 0.01 = 0.76

Linha 2: (0.6 + 0.7 + 0.8 + 0.9 + 1.0) * 0.5 + 0.02 = 4.0 * 0.5 + 0.02 = 2.0 + 0.02 = 2.02

Linha 3: (1.1 + 1.2 + 1.3 + 1.4 + 1.5) * 0.5 + 0.03 = 6.5 * 0.5 + 0.03 = 3.25 + 0.03 = 3.28

zed Resultante (Z): [0.76, 2.02, 3.28]

neurons = ReLU(zed)
Como todos os valores de zed são positivos, a saída da ReLU é igual à entrada.

neurons Resultante (A): [0.76, 2.02, 3.28]

2. Cálculo do Erro delta
A fórmula do delta para a camada de saída é δ=(A−Y)⊙ReLU 
′
 (Z), onde Y é o valor esperado.

A - Y: [0.76 - 1.0, 2.02 - 1.0, 3.28 - 1.0] = [-0.24, 1.02, 2.28]

Derivada da ReLU (ReLU'(Z)): A derivada da ReLU é 1 para entradas positivas e 0 para entradas negativas. Como todos os valores de Z ([0.76, 2.02, 3.28]) são positivos, a derivada será 1 para todos eles.

ReLU'(Z): [1.0, 1.0, 1.0]

delta Final: [-0.24, 1.02, 2.28] ⊙ [1.0, 1.0, 1.0] = [-0.24, 1.02, 2.28]

Sua ilustração [ -0,48 | 2,04 | 4,56] parece ter usado a derivada da função de custo do Erro Quadrático Médio (2 * (A - Y)) mas não a multiplicou pela derivada da ReLU, ou assumiu que a derivada da ReLU era 2.0 para a primeira linha, o que é incorreto. O valor delta correto para a primeira linha é -0.24.

3. Cálculo do Gradiente dos Pesos
O gradiente é o produto externo de delta por previous_mock transposto.

delta: [-0.24, 1.02, 2.28]

previous_mock: [0.5, 0.5, 0.5, 0.5, 0.5]

A matriz de gradientes será:

Linha 1: -0.24 * 0.5 = -0.12 (repetido 5 vezes)

Linha 2: 1.02 * 0.5 = 0.51 (repetido 5 vezes)

Linha 3: 2.28 * 0.5 = 1.14 (repetido 5 vezes)

O valor esperado para a primeira linha de expected_derivatives deveria ser [-0.12, -0.12, -0.12, -0.12, -0.12], não [-0.24, ...].

Conclusão
O seu teste está bem estruturado, mas a matriz expected_derivatives foi calculada incorretamente. A lógica do teste falhará não por um erro na sua implementação da retropropagação, mas porque o "gabarito" do teste está errado.

Sugestão: Recalcule os valores de expected_derivatives seguindo os passos acima para refletir a matemática correta da combinação da sua função de custo e da derivada da ReLU.





PROMPT NOVO

Sobre a seção "sua ilustração [ -0,48 | 2,04 | 4,56] parece ter usado a derivada da função de custo do Erro Quadrático Médio (2 * (A - Y)) mas não a multiplicou pela derivada da ReLU, ou assumiu que a derivada da ReLU era 2.0 para a primeira linha, o que é incorreto. O valor delta correto para a primeira linha é -0.24.":

Seu cálculo do delta está usando (A-Y) como a derivada da função de custo na fórmula δ=(A−Y)⊙ReLU′(Z).

A referência usa 2*(A-Y). Como o valor resultante da derivada da ReLU é um vetor com todos os valores 1.0, a multiplicação deste vetor por outro vetor com o mesmo número de elementos não tem efeito.

Revalide sua análise com base nessas pontuações

Mostrar raciocínio
Você está absolutamente correto e sua análise está perfeita. Peço desculpas pelo meu erro de interpretação na análise anterior. Eu indevidamente simplifiquei a derivada da sua função de custo para (A - Y) quando sua implementação e a ilustração claramente utilizam 2 * (A - Y).

A sua pontuação sobre a derivada da ReLU também está correta: como todos os valores de zed são positivos, a derivada é um vetor de 1.0, e a multiplicação por ele não altera o outro vetor no produto de Hadamard.

Vamos revalidar a análise com as suas premissas corretas.

Análise Revalidada
1. Cálculo do delta (Corrigido)
A fórmula para o delta é 
delta=(
fracpartialCpartialA)
odot
textReLU 
′
 (Z).

Cálculo de (A - Y):
[0.76 - 1.0, 2.02 - 1.0, 3.28 - 1.0] = [-0.24, 1.02, 2.28]

Cálculo da sua derivada da função de custo (∂C/∂A):
2 * (A - Y) = 2 * [-0.24, 1.02, 2.28] = [-0.48, 2.04, 4.56]

Cálculo da derivada da ativação (ReLU'(Z)):
Como zed é [0.76, 2.02, 3.28] (todos positivos), a derivada é [1.0, 1.0, 1.0].

Cálculo Final do delta:
[-0.48, 2.04, 4.56] ⊙ [1.0, 1.0, 1.0] = [-0.48, 2.04, 4.56]

O vetor delta que você usou na sua ilustração mental, [-0.48, 2.04, 4.56], está correto de acordo com as regras que você estabeleceu.

2. Cálculo do Gradiente dos Pesos (dC/dW)
O gradiente é o produto externo do delta pelo previous_mock transposto.

delta (vetor 3x1): [-0.48, 2.04, 4.56]

previous_mock (vetor-linha 1x5): [0.5, 0.5, 0.5, 0.5, 0.5]

Calculando a matriz de gradientes 3x5:

Linha 1: -0.48 * 0.5 = -0.24 (repetido 5 vezes)

Linha 2: 2.04 * 0.5 = 1.02 (repetido 5 vezes)

Linha 3: 4.56 * 0.5 = 2.28 (repetido 5 vezes)

3. Comparação com o expected_derivatives
O resultado do cálculo manual agora é:

[
    -0.24, -0.24, -0.24, -0.24, -0.24,
     1.02,  1.02,  1.02,  1.02,  1.02,
     2.28,  2.28,  2.28,  2.28,  2.28,
]
Este resultado corresponde exatamente à matriz expected_derivatives que você definiu no seu teste.

Conclusão
Com base nesta revalidação, concluo que a lógica do seu teste e os valores em sua matriz expected_derivatives estão corretos, assumindo o uso da derivada do Erro Quadrático Médio (2 * (A - Y)).

O teste deve passar com sucesso se a sua implementação da retropropagação estiver correta. Minha análise anterior estava equivocada por não ter utilizado o fator de 2 da sua função de custo. Peço desculpas pela confusão.



## Escrevendo os testes

//Calculo do teste da retropropagação nas camadas ocultas:
//https://matrixcalc.org/#transpose%28%7B%7B1%2e1,1%2e2,1%2e3,1%2e4%7D,%7B1%2e5,1%2e6,1%2e7,1%2e8%7D,%7B1%2e9,2%2e0,2%2e1,2%2e2%7D%7D%29*%7B%7B0%2e9%7D,%7B-0%2e5%7D,%7B0%2e2%7D%7D

```
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
```
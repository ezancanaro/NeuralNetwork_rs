Aqui eu cheguei na minha primeira dificuldade com a compreensão da rede neural. Por que a matriz de pesos deve ser transposta para a propagação dos valores em neurônios? 
Nesse ponto, precisamos entender porque utilizamos a representação de matrizes para os componentes da rede neural.

A primeira intuição está na relação entre o produto escalar (*dot product*) e a multiplicação de matrizes. 
Considere o caso de um único neurônio da camada. O valor desse neurônio é dado pela multiplicação do valor de entrada (neurônio da camada anterior), pelo peso atribuído (w * v). Se ele recebe sinais de múltiplos neurônios na camada anterior, seu valor é dado pela soma ponderada de suas entradas (valor multiplicado pelo peso): sum(w_i * v_i).

Se representamos os neurônios da camada de entrada no formato de um vetor e_n, e ordenamos os pesos associados ao neurônio saida (s_n) como um vetor w_n, então o valor do neurônio é dado pelo produto escalar desses vetores:


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

 
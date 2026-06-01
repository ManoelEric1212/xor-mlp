# Rede Neural MLP para Aprendizado da Porta Lógica XOR

## Base de dados

A base `X` representa a tabela verdade da porta lógica XOR.

```octave
X = [
  0, 0, 0;
  0, 1, 1;
  1, 0, 1;
  1, 1, 0
];
```

A porta XOR possui a seguinte tabela verdade:

| Entrada $x_1$ | Entrada $x_2$ | Saída $y$ |
| :-----------: | :-----------: | :-------: |
|       0       |       0       |     0     |
|       0       |       1       |     1     |
|       1       |       0       |     1     |
|       1       |       1       |     0     |

A porta XOR não é linearmente separável. Por isso, um perceptron simples não é suficiente para resolver esse problema. Para resolver esse caso, utiliza-se uma rede neural MLP com uma camada oculta.

---

## Arquitetura da MLP

A rede neural utilizada possui a seguinte estrutura:

- 2 neurônios na camada de entrada;
- 2 neurônios na camada oculta;
- 1 neurônio na camada de saída;
- bias na camada oculta;
- bias na camada de saída;
- função de ativação sigmoide;
- treinamento com backpropagation.

A arquitetura pode ser representada como:

Os pesos foram inicializados da seguinte forma:

```octave
pesosEntradaOculta = rand(3, 2) * 2 - 1;
pesosOcultaSaida = rand(3, 1) * 2 - 1;
```

A matriz `pesosEntradaOculta` possui dimensão `3 x 2`, pois contém:

- 1 linha para o bias;
- 2 linhas para as entradas `x1` e `x2`;
- 2 colunas, uma para cada neurônio da camada oculta.

A matriz `pesosOcultaSaida` possui dimensão `3 x 1`, pois contém:

- 1 linha para o bias;
- 2 linhas para as saídas dos neurônios ocultos;
- 1 coluna para o neurônio de saída.

---

## Função de ativação

A função de ativação utilizada foi a sigmoide:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

A derivada da função sigmoide é:

$$
f'(x) = f(x)(1 - f(x))
$$

Essa derivada é utilizada durante o backpropagation.

---

## Propagação direta

A saída da camada oculta é calculada por:

$$
z_{oculta} = W_{entrada,oculta}^{T} \cdot X
$$

$$
h = f(z_{oculta})
$$


A saída final da rede, na camada de saída é calculada por:

$$
z_{saida} = W_{oculta,saida}^{T} \cdot H
$$

$$
\hat{y} = f(z_{saida})
$$

---

## Cálculo do erro

O erro é calculado pela diferença entre a saída desejada e a saída obtida:

$$
erro = y - \hat{y}
$$

Também é calculado o erro quadrático:

$$
EQ = erro^2
$$

E o erro médio da época:

$$
erroMedio = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2
$$

---

## Backpropagation

O backpropagation ajusta os pesos da rede neural com base no erro obtido.

### Delta da camada de saída

O delta da camada de saída é calculado por:

$$
\delta_{saida} = erro \cdot \hat{y}(1 - \hat{y})
$$


### Delta da camada oculta

O delta da camada oculta considera o erro propagado da saída para os neurônios ocultos:

$$
\delta_{oculta} =
(W_{oculta,saida} \cdot \delta_{saida}) \cdot h(1 - h)
$$

Lembrando que o bias não recebe o erro retropropagado.

---

## Atualização dos pesos

Os pesos são atualizados usando a taxa de aprendizagem $\eta$:

$$
W_{novo} = W_{atual} + \eta \cdot \delta \cdot entrada
$$

Os pesos entre a camada oculta e a saída são atualizados por:

```octave
pesosOcultaSaida = pesosOcultaSaida + ...
                   taxaAprendizagem * deltaSaida * saidaOcultaComBias;
```

Os pesos entre a entrada e a camada oculta são atualizados por:

```octave
pesosEntradaOculta = pesosEntradaOculta + ...
                     taxaAprendizagem * entrada * deltaOculta';
```

---

## Parâmetros utilizados

| Parâmetro               | Valor    |
| :---------------------- | :------- |
| Taxa de aprendizagem    | `0.7`    |
| Número máximo de épocas | `1000`  |
| Erro mínimo             | `0.01`   |
| Neurônios de entrada    | `2`      |
| Neurônios ocultos       | `2`      |
| Neurônio de saída       | `1`      |
| Função de ativação      | Sigmoide |

---

## Critério de parada

O treinamento é executado até que uma das condições seja satisfeita:

1. O número máximo de épocas seja atingido;
2. O erro médio seja menor que o erro mínimo definido.


---

## Teste da rede neural

Após o treinamento, a rede é testada com todas as combinações da porta XOR. A saída numérica da rede é convertida para uma saída binária usando o limiar `0.5`:

$$
saida =
\begin{cases}
1, & \text{se } \hat{y} \geq 0.5 \\
0, & \text{se } \hat{y} < 0.5
\end{cases}
$$


---

## Resultados esperados

Após o treinamento, espera-se que a rede neural aprenda o comportamento da porta XOR:

| Entrada $x_1$ | Entrada $x_2$ | Saída esperada | Saída da rede |
| :-----------: | :-----------: | :------------: | :-----------: |
|       0       |       0       |       0        | próximo de 0  |
|       0       |       1       |       1        | próximo de 1  |
|       1       |       0       |       1        | próximo de 1  |
|       1       |       1       |       0        | próximo de 0  |

Um exemplo de saída obtida foi:

```text
Teste da rede neural para XOR:
0 XOR 0 = 0.1026 -> 0
0 XOR 1 = 0.8948 -> 1
1 XOR 0 = 0.9001 -> 1
1 XOR 1 = 0.0875 -> 0
```

Os valores numéricos podem variar de acordo com a inicialização dos pesos, a taxa de aprendizagem e o número de épocas.

---

## Conclusão

A rede neural MLP conseguiu resolver o problema da porta lógica XOR, que não pode ser resolvido por um perceptron simples.

Isso ocorre porque a MLP possui uma camada oculta, permitindo que a rede aprenda relações não lineares entre as entradas.

Portanto, a arquitetura com dois neurônios na camada oculta e um neurônio na camada de saída é suficiente para aprender o comportamento da porta XOR.

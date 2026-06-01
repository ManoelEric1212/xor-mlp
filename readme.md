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

Para cada amostra de entrada, é adicionado o bias:

```octave
entrada = [1; entradas(j, :)'];
```

A saída da camada oculta é calculada por:

$$
z_{oculta} = W_{entrada,oculta}^{T}x
$$

$$
h = f(z_{oculta})
$$

No código:

```octave
somaOculta = pesosEntradaOculta' * entrada;
saidaOculta = 1 ./ (1 + exp(-somaOculta));
```

Em seguida, adiciona-se o bias à saída da camada oculta:

```octave
saidaOcultaComBias = [1; saidaOculta];
```

A saída final da rede é calculada por:

$$
z_{saida} = W_{oculta,saida}^{T}h
$$

$$
\hat{y} = f(z_{saida})
$$

No código:

```octave
somaSaida = pesosOcultaSaida' * saidaOcultaComBias;
saidaObtida = 1 / (1 + exp(-somaSaida));
```

---

## Cálculo do erro

O erro é calculado pela diferença entre a saída desejada e a saída obtida:

$$
erro = y - \hat{y}
$$

No Octave:

```octave
erro = saidaDesejada - saidaObtida;
```

Também é calculado o erro quadrático:

$$
EQ = erro^2
$$

E o erro médio da época:

$$
erroMedio = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2
$$

No código:

```octave
erroQuadratico = erroQuadratico + erro^2;
erroMedio = erroQuadratico / size(entradas, 1);
```

---

## Backpropagation

O backpropagation ajusta os pesos da rede neural com base no erro obtido.

### Delta da camada de saída

O delta da camada de saída é calculado por:

$$
\delta_{saida} = erro \cdot \hat{y}(1 - \hat{y})
$$

No código:

```octave
deltaSaida = erro * saidaObtida * (1 - saidaObtida);
```

### Delta da camada oculta

O delta da camada oculta considera o erro propagado da saída para os neurônios ocultos:

$$
\delta_{oculta} =
(W_{oculta,saida} \cdot \delta_{saida}) \cdot h(1 - h)
$$

No código:

```octave
deltaOculta = (pesosOcultaSaida(2:end) * deltaSaida) ...
              .* saidaOculta .* (1 - saidaOculta);
```

O trecho `pesosOcultaSaida(2:end)` ignora o peso do bias, pois o bias não recebe erro retropropagado.

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
| Número máximo de épocas | `10000`  |
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

```octave
if erroMedio < erroMinimo
    fprintf("Treinamento finalizado na epoca %d\n", epoca);
    break;
end
```

---

## Teste da rede neural

Após o treinamento, a rede é testada com todas as combinações da porta XOR:

```octave
fprintf("\nTeste da rede neural para XOR:\n");

for j = 1:size(entradas, 1)

    entrada = [1; entradas(j, :)'];

    somaOculta = pesosEntradaOculta' * entrada;
    saidaOculta = 1 ./ (1 + exp(-somaOculta));

    saidaOcultaComBias = [1; saidaOculta];

    somaSaida = pesosOcultaSaida' * saidaOcultaComBias;
    saidaObtida = 1 / (1 + exp(-somaSaida));

    saidaBinaria = saidaObtida >= 0.5;

    fprintf("%d XOR %d = %.4f -> %d\n", ...
            entradas(j, 1), entradas(j, 2), saidaObtida, saidaBinaria);

end
```

A saída numérica da rede é convertida para uma saída binária usando o limiar `0.5`:

$$
saida =
\begin{cases}
1, & \text{se } \hat{y} \geq 0.5 \\
0, & \text{se } \hat{y} < 0.5
\end{cases}
$$

No código:

```octave
saidaBinaria = saidaObtida >= 0.5;
```

---

## Entrada do usuário

Após o treinamento, o programa solicita ao usuário dois valores de entrada:

```octave
x1 = input("Digite o valor de x1, 0 ou 1: ");
x2 = input("Digite o valor de x2, 0 ou 1: ");
```

Esses valores são enviados para a rede neural treinada:

```octave
entradaUsuario = [1; x1; x2];

somaOculta = pesosEntradaOculta' * entradaUsuario;
saidaOculta = 1 ./ (1 + exp(-somaOculta));

saidaOcultaComBias = [1; saidaOculta];

somaSaida = pesosOcultaSaida' * saidaOcultaComBias;
saidaRede = 1 / (1 + exp(-somaSaida));

saidaFinal = saidaRede >= 0.5;
```

Por fim, o programa exibe a saída numérica e a saída binária da rede:

```octave
fprintf("\nSaida numerica da rede: %.4f\n", saidaRede);
fprintf("Saida binaria da rede: %d\n", saidaFinal);
```

---

## Resultados esperados

Após o treinamento, espera-se que a rede neural aprenda o comportamento da porta XOR:

| Entrada $x_1$ | Entrada $x_2$ | Saída esperada | Saída da rede |
| :-----------: | :-----------: | :------------: | :-----------: |
|       0       |       0       |       0        | próximo de 0  |
|       0       |       1       |       1        | próximo de 1  |
|       1       |       0       |       1        | próximo de 1  |
|       1       |       1       |       0        | próximo de 0  |

Um exemplo de saída possível é:

```text
Teste da rede neural para XOR:

0 XOR 0 = 0.08 -> 0
0 XOR 1 = 0.91 -> 1
1 XOR 0 = 0.91 -> 1
1 XOR 1 = 0.10 -> 0
```

Os valores numéricos podem variar de acordo com a inicialização dos pesos, a taxa de aprendizagem e o número de épocas.

Como foi utilizada a instrução:

```octave
rand("seed", 1);
```

a geração dos pesos iniciais tende a ser repetível, fazendo com que o resultado seja semelhante em diferentes execuções.

---

## Conclusão

A rede neural MLP conseguiu resolver o problema da porta lógica XOR, que não pode ser resolvido por um perceptron simples.

Isso ocorre porque a MLP possui uma camada oculta, permitindo que a rede aprenda relações não lineares entre as entradas.

Portanto, a arquitetura com dois neurônios na camada oculta e um neurônio na camada de saída é suficiente para aprender o comportamento da porta XOR.

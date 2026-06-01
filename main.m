clc;
clear;

X = [
  0, 0, 0;
  0, 1, 1;
  1, 0, 1;
  1, 1, 0
];

entradas = X(:, 1:2);
saidasDesejadas = X(:, 3);
taxaAprendizagem = 0.7;
epocas = 10000;
erroMinimo = 0.01;

% Para repetir sempre o mesmo resultado
rand("seed", 1);


pesosEntradaOculta = rand(3, 2) * 2 - 1;
pesosOcultaSaida = rand(3, 1) * 2 - 1;


for epoca = 1:epocas
    erroQuadratico = 0;
    for j = 1:size(entradas, 1)
        entrada = [1; entradas(j, :)'];
        saidaDesejada = saidasDesejadas(j);
        somaOculta = pesosEntradaOculta' * entrada;
        saidaOculta = 1 ./ (1 + exp(-somaOculta));
        saidaOcultaComBias = [1; saidaOculta];
        somaSaida = pesosOcultaSaida' * saidaOcultaComBias;
        saidaObtida = 1 / (1 + exp(-somaSaida));
        erro = saidaDesejada - saidaObtida;
        erroQuadratico = erroQuadratico + erro^2;


        deltaSaida = erro * saidaObtida * (1 - saidaObtida);
        deltaOculta = (pesosOcultaSaida(2:end) * deltaSaida) .* saidaOculta .* (1 - saidaOculta);
        pesosOcultaSaida = pesosOcultaSaida + taxaAprendizagem * deltaSaida * saidaOcultaComBias;
        pesosEntradaOculta = pesosEntradaOculta +taxaAprendizagem * entrada * deltaOculta';
    end
    erroMedio = erroQuadratico / size(entradas, 1);
    if erroMedio < erroMinimo
        fprintf("Treinamento finalizado na epoca %d\n", epoca);
        break;
    end
end



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


fprintf("\nAgora informe os valores de entrada.\n");

x1 = input("Digite o valor de x1, 0 ou 1: ");
x2 = input("Digite o valor de x2, 0 ou 1: ");

entradaUsuario = [1; x1; x2];

somaOculta = pesosEntradaOculta' * entradaUsuario;
saidaOculta = 1 ./ (1 + exp(-somaOculta));
saidaOcultaComBias = [1; saidaOculta];

somaSaida = pesosOcultaSaida' * saidaOcultaComBias;
saidaRede = 1 / (1 + exp(-somaSaida));

saidaFinal = saidaRede >= 0.5;

fprintf("\nSaida numerica da rede: %.4f\n", saidaRede);
fprintf("Saida binaria da rede: %d\n", saidaFinal);

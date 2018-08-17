# Rede perceptron(1-C)
Características:
- Uma camada de saída contendo c neurônios.
- Sem camadas ocultas.
- Normalização das bases de dados.
- Datasets
	- Iris
	- Artificial
- Funções de ativação
	- Linear
	- Sigmoide

## Realização para base Artificial
Parâmetros:
- Número de épocas: 400
- Taxa de aprendizagem: 0.1
- Proporção de treinamento: 80%
- Proporção de testes: 20%
- Função de ativação: Linear
- Total de realizações:20

### Resultados:


    Desvio padrão:  0.01699673171197594
    Acurácia:  99.33333333333334 %
    Melhor taxa de acerto:  100%
    Melhor matriz w:
    [ 0.1        -0.61573976  0.46695825]
    [-0.1         0.34073657 -0.66883564]
    [ 0.2         0.15646026  0.15226348]

![enter image description here](https://lh3.googleusercontent.com/1Xo3dVvFJjl99AskaWc8_wszZxGTMJplsFWsJO9EalK6SPYQr1kZMT37X-QchNdBjHMfpf_Rwrs)

## Realização para base Iris
Parâmetros:
- Número de épocas: 400
- Taxa de aprendizagem: 0.1
- Proporção de treinamento: 80%
- Proporção de testes: 20%
- Função de ativação: Linear
- Total de realizações: 20

### Resultados
### Treinamento com todos os atributos:

    Desvio padrão:  0.08225975119502045
    Acurácia:  69.66666666666664 %
    Melhor taxa de acerto: 86 %
    Melhor matriz W:
    [-1.38777878e-16 -1.26114056e-01  4.25584896e-01 -2.57867317e-01 -7.25301872e-02]
    [-1.00000000e-01 -1.33075337e-01 -3.67519048e-01  6.08955045e-01 -5.64501734e-01]
    [ 2.40000000e+00 -1.47024003e-01 -7.72877114e-01  2.49995604e+00 1.68282887e+00]

### Treinamento com apenas 2 atributos:

    Desvio padrão:  0.08685876147196923
    Acurácia:  66.33 %
    Melhor taxa de acerto:  80%
    Melhor matriz w:
    [-0.1        -0.0087332  -0.40903056]
    [ 0.1         0.79112821 -0.60024686]
    [ 1.6         0.71695543  1.64021777]

![enter image description here](https://lh3.googleusercontent.com/xYHkPuxLEhMuCi0HrSZvfXpjpr5dTKwqDVcA5SSXP_vXc0FXaRdTZ3cpEfVJ0lCn0LRq-fzie44)
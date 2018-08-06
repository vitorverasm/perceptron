import numpy as np


class Perceptron():

    def __init__(self, data, functions, neurons=3, proportion=0.8, eta=0.001, epochs=200):
        self.data = data  # Base de dados
        self.functions = functions
        self.proportion = proportion  # treina com 80% dos dados e testa com 20% dos dados
        self.eta = eta  # Taxa de aprendizagem
        self.epochs = epochs  # Número máximo de épocas
        self.neurons = neurons
        self.bias = -1.0  # x0
        self.theta = -1.0  # w0
        self.data = self.insertBias()  # insere valor do x0 na base de dados
        self.w = self.initW()  # incializa w (aleatório com theta w0)

    # Retorna a ultima coluna de certo x
    def desired(self, x):
        return int(x[x.size - 1])

    # Insere o valor do x0(bias) na base de dados recebida
    def insertBias(self):
        d = []
        for i in range(len(self.data)):
            d.append(np.insert(self.data[i], 0, self.bias))  # insere o valor de x0 para todos os padrões
        d = np.asarray(d)
        return d

    # Inicializa a matrix w com mesmo numero de colunas da base e com c linhas(número de neuronios)
    def initW(self):
        matrix = np.random.rand(self.neurons, self.data.shape[1])
        matrix[:, 0] = self.theta
        return matrix

    # Imprime informações gerais sobre o modelo
    def printInfo(self):
        print("Informações: \n")
        print("Dados:", self.data)
        print("Proporção de treinamento/testes:", self.proportion)
        print("Taxa de aprendizagem:", self.eta)
        print("Número de épocas: ", self.epochs)
        print("Vetor w inicial: \n", self.w)
        print("Funções de ativação:", self.getFunctionsNames())

    # Calcula o produto interno w.xT
    def dotProduct(self, x):
        xT = np.asarray([x]).T
        return np.dot(self.w, xT)

    # Retorna uma lista com as saidas dos neuronios
    def y(self, x):
        y = []
        u = self.dotProduct(x)
        for i in range(self.neurons):
            y.append(self.functions[i][1](u[i][0]))
        return y

    # Retorna uma lista de erros
    def error(self, x):
        e = []
        y = self.y(x)
        for i in range(self.neurons):
            e.append(self.desired(x) - y[i])
        return e

    # Retorna uma lista com os nomes das funções de ativação
    def getFunctionsNames(self):
        fns = []
        for i in self.functions:
            fns.append(i[0])
        return fns

    # TODO treinamento
    def training(self):
        pass

    # TODO testes
    def test(self):
        pass

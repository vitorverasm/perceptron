import numpy as np


class Perceptron():

    def __init__(self, data, function, neurons=3, proportion=0.8, eta=0.1, epochs=400):
        self.data = data  # Base de dados
        self.functionName = function[0]  # Recebe o nome da função de ativação
        self.function = function[1]  # Recebe a função de ativação
        self.proportion = proportion  # treina com 80% dos dados e testa com 20% dos dados
        self.eta = eta  # Taxa de aprendizagem
        self.epochs = epochs  # Número máximo de épocas
        self.neurons = neurons
        self.bias = -1.0  # x0
        self.theta = -1.0  # w0
        self.data = self.insertBias()  # insere valor do x0 na base de dados
        self.w = self.initW()  # incializa w (aleatório com theta w0)

    # Retorna a ultima coluna de certo x com o valor de classe desejado
    # Filtra a classe pelo index do neuronio
    def desired(self, x):
        d = []
        label = x[x.size - 1]
        for index in range(self.neurons):
            if label == index:
                d.append(1)
            else:
                d.append(0)
        return d

    # Insere o valor do bias(x0) na base de dados recebida
    def insertBias(self):
        d = []
        for i in range(len(self.data)):
            d.append(np.insert(self.data[i], 0, self.bias))  # insere o valor de x0 para todos os padrões
        d = np.asarray(d)
        return d

    # Inicializa a matrix w com mesmo numero de colunas da base e com c linhas(número de neuronios)
    def initW(self):
        matrix = np.random.rand(self.neurons, (self.data.shape[1] - 1))
        matrix[:, 0] = self.theta
        return matrix

    # Imprime informações gerais sobre o modelo da rede
    def printInfo(self):
        print("Informações: \n")
        print("Dados:", self.data)
        print("Proporção de treinamento/testes:", self.proportion)
        print("Taxa de aprendizagem:", self.eta)
        print("Número de épocas: ", self.epochs)
        print("Vetor w inicial: \n", self.w)
        print("Função de ativação:", self.functionName)

    # Calcula o produto interno w[i]T.x
    # Onde, i é o index do neurônio
    def dotProduct(self, pattern):
        x = pattern[0:-1]
        w = self.w
        u = []
        for index in range(self.neurons):
            u.append(np.dot(w[index], x))
        return u

    # Retorna uma lista com as saidas dos neuronios
    def y(self, x):
        u = self.dotProduct(x)
        y = []
        for index in range(self.neurons):
            y.append(self.function(u[index]))
        return y

    # REGRA DE APRENDIZAGEM / AJUSTE DO VETOR W
    # w(t+1)=w(t) + (taxa_aprendizagem * erro_iteração)*x(t)
    def adjust(self, pattern, error):
        x = pattern[0:-1]
        for index in range(self.neurons):
            self.w[index] = self.w[index] + (self.eta) * (error[index]) * x

    # Retorna uma lista de erros de cada neuronio
    def error(self, x):
        e = []
        y = self.y(x)
        d = self.desired(x)
        for index in range(self.neurons):
            e.append(d[index] - y[index])
        return np.array(e)

    def training(self):
        data = self.data[0: int(len(self.data) * self.proportion)]  # utiliza apenas a proporção certa dos dados
        i = 1
        while i < self.epochs:
            np.random.shuffle(data)  # shuffle entre épocas
            for x in data:
                error = self.error(x)
                self.adjust(x, error)
            i += 1
        return self.w

    # TESTES
    def test(self):
        data = self.data[int(len(self.data) * self.proportion):]  # utiliza apenas a proporção certa dos dados
        acc = []
        for x in data:
            e = sum(self.error(x))
            if e == 0:
                acc.append(1)
            else:
                acc.append(0)

        return sum(acc) / len(data)

    # REALIZAÇÃO
    # Faz uma realização completa, que consiste em:
    #   - Treino
    #   - Testes
    def execution(self, times):
        acc_tx = []  # lista que salva a taxa de acerto de cada realização
        q=0
        dataToPlot = self.data
        wToPlot = self.w
        print("### PERCEPTRON SIMPLES ###")
        print("PARÂMETROS: ")
        self.printInfo()
        print("Total de realizações: ", times, "\n")

        for i in range(1, times + 1):
            print("### REALIZAÇÃO ", i, "###")
            np.random.shuffle(self.data)  # shuffle entre realizações
            self.w = self.initW()  # reseta o vetor de pesos entre realizações
            print("### FASE DE TREINAMENTO ###")
            w = self.training()
            print("Vetor W final: ", w)
            print("### FASE DE TESTES ###")
            tx = self.test()
            print("Taxa de acerto: ", tx, "\n")
            acc_tx.append(tx)
            if q<tx:
                q=tx
                dataToPlot = self.data
                wToPlot = self.w

        accuracy = (sum(acc_tx) / times)  # acurácia entre [0,1]
        # Cálculo do desvio padrão
        dp = self.standardDeviation(accuracy, acc_tx)
        accuracy *= 100  # acurácia em porcentagem
        print("DESVIO PADRÃO: ", dp)
        print("ACURÁCIA: ", accuracy)
        print("### FIM PERCEPTRON ###")
        return [q, dataToPlot, wToPlot]

    def standardDeviation(self, mean, list):
        aux = []
        for i in list:
            aux.append((i - mean) ** 2)
        d = np.sqrt(sum(aux) / len(list))
        return d

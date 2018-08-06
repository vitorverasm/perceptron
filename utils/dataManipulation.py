import pandas as pd
import numpy as np
import random


class DataManipulation():
    # Construtor
    def __init__(self, path, type):
        self.path = path
        # Define o tratamento dependendo do dataset passado
        if type == 0:
            # Para a iris os dados já estão gerados
            # basta ler e adequar
            self.data = np.array(self.readData())
            self.generateIris()
        else:
            # Já para o artificial é necessário ainda gerar os dados
            # Para após ler
            self.generateArtificial(40)
            self.data = np.array(self.readData())
        # Embaralha os dados
        np.random.shuffle(self.data)

    # Lê o arquivo e retorna um DataFrame(pandas)
    def readData(self):
        data = pd.read_csv(self.path, delimiter=',', header=None)
        return data

    # Função auxiliar para o banco de dados Iris.data
    # Torna as classes(labels) binárias(setosa contra outras)
    def generateIris(self):
        for x in self.data:
            if x[len(x) - 1] == 'Iris-setosa':
                x[len(x) - 1] = 0
            else:
                x[len(x) - 1] = 1

    # Retorna os dados
    def getData(self):
        return self.data

    # Função de geração da base de dados artificial I
    # De acordo com as caracteristicas de proporção(3/4 e 1/4)
    # gera os valores da classe 0 proximos de 0
    # gera os valores da classe 1 proximos de 1
    def generateArtificial(self, amount):
        data = []
        for i in range(int(amount * 3 / 4)):
            a = random.uniform(0, 0.5)
            b = random.uniform(0, 0.5)
            data.append([a, b, 0])
        for j in range(int(amount * 1 / 4)):
            a = 1 - (random.uniform(0, 0.5))
            b = 1 - (random.uniform(0, 0.5))
            data.append([a, b, 1])
        # Escreve no arquivo
        df = pd.DataFrame(data)
        df.to_csv(self.path, index=False, header=None)

    # Normaliza um dataframe
    def normalize(self, df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

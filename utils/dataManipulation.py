import pandas as pd
import numpy as np
from random import uniform

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
            self.generateArtificial(50)
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
            if x[len(x) - 1] == 'Iris-versicolor':
                x[len(x) - 1] = 1
            if x[len(x) - 1] == 'Iris-virginica':
                x[len(x) - 1] = 2

        df = pd.DataFrame(self.data)
        df = self.normalize(df)
        df.to_csv(self.path, index=False, header=None)

    # Retorna os dados
    def getData(self):
        return self.data

    # Função de geração da base de dados artificial
    def generateArtificial(self, n):
        data = []
        default = ([0, 3, 0], [1, 1, 1], [2, 3, 2])
        for x1, x2, label in default:
            for i in range(n):
                data.append([x1 + uniform(0.0, 1.0), x2 + uniform(0.0, 1.0), label])
        # Escreve no arquivo
        df = pd.DataFrame(data)
        df = self.normalize(df)
        df.to_csv(self.path, index=False, header=None)

    # Normaliza um dataframe
    def normalize(self, df):
        result = df.copy()

        for feature_name in range(len(df.columns) - 1):
            max_value = df[df.columns[feature_name]].max()
            min_value = df[df.columns[feature_name]].min()
            result[df.columns[feature_name]] = (df[df.columns[feature_name]] - min_value) / (max_value - min_value)
        return result

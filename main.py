from src.perceptron import Perceptron
from utils.functions import step_fn
from utils.dataManipulation import DataManipulation
import utils.plot as plotData
# CAMINHOS DAS BASES DE DADOS
iris_path = './samples/iris.data'
art_path = './samples/artificial.data'

# PARAMETROS
problem = 'iris'
realizacoes = 20
fn = step_fn()
todosAtributos = False

if problem == 'iris':
    dm = DataManipulation(iris_path, 0)
    if todosAtributos:
        data = dm.getData()  # Base iris com apenas todos atributos
    else:
        data = [[p[2:]] for p in dm.getData()]  # Base iris com apenas 2 atributos
else:
    dm = DataManipulation(art_path, 1)
    data = dm.getData()  # Base artificial

p = Perceptron(data, fn)
r = p.execution(realizacoes)
# Plot
bestAcc = r[0]
bestAccData = r[1]
bestW = r[2]
print()
print('### Informações do plot ###')
print('Melhor taxa de acerto: ', bestAcc)
print('Melhor vetor w: ', bestW)
print('#############################')

if (problem == 'iris' or problem == 'artificial') and todosAtributos == False:
    plotData.plotPatterns(bestAccData)
    plotData.plot(bestAccData, bestW[0])
    plotData.plot(bestAccData, bestW[1])
    plotData.plot(bestAccData, bestW[2]).show()


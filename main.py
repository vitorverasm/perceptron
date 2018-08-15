from src.perceptron import Perceptron
from utils.functions import step_fn
from utils.dataManipulation import DataManipulation
import utils.plot as plotData

iris_path = './samples/iris.data'
art_path = './samples/artificial.data'
fn = step_fn()
dm = DataManipulation(iris_path, 0)
p = Perceptron(dm.getData(), fn)
p.execution(10)
# plotData.plotPatterns(p.data, "bla bla")
plotData.plot2(p.data)
# plotData.plot(p.data, p.w[0])
# plotData.plot(p.data, p.w[1])
# plotData.plot(p.data, p.w[2]).show()
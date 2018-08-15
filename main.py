import numpy as np
from src.perceptron import Perceptron
from utils.functions import step_fn
from utils.dataManipulation import DataManipulation

iris_path = './samples/iris.data'
art_path = './samples/artificial.data'
fn = step_fn()
dm = DataManipulation(iris_path, 0)
p = Perceptron(dm.getData(), fn)
p.execution(10)

import numpy as np
from src.perceptron import Perceptron
from utils.functions import step_fn
from utils.dataManipulation import DataManipulation

iris_path = './samples/iris.data'
fn = [step_fn(), step_fn(), step_fn()]
dm = DataManipulation(iris_path, 0)

p = Perceptron(dm.getData(), fn)
p.printInfo()

import numpy as np
from src.perceptron import Perceptron
from utils.functions import step_fn
data = np.asarray([[1,2,3,0],[4,5,6,0],[7,8,9,0]])
fn = [step_fn(), step_fn(), step_fn()]

p = Perceptron(data, fn)
p.printInfo()
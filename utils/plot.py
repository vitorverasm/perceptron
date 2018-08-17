import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plota os padrões de dados da base
def plotPatterns(data, title="Iris Dataset Plot"):
    plt.title(title)
    for x in data:
        if x[x.size -1] == 0:
            color = 'b'
            marker = 'o'
        elif x[x.size -1] == 1:
            color = 'r'
            marker = 's'
        else:
            color = 'g'
            marker = '^'
        plt.plot(x[1], x[2], color=color, marker=marker, label=x[x.size -1])
    return plt

# Plota a reta de separação para o w passado
def plot(data, W):
    p1 = []
    p2 = []
    data = [[p[1:-1]] for p in data]
    for x1 in np.linspace(np.amin(data), np.amax(data)):
        p1.append(x1)
        x2 = -(W[1] / W[2]) * x1 + (W[0] / W[2])
        p2.append(x2)
    plt.plot(p1, p2, '-')
    return plt


# Plot da base de dados da iris(em pares)
def plot2(data):
    d = []
    for x in data:
        d.append(x[1:])
    df = pd.DataFrame(d, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    plt.title("Iris Data Set")
    g = sns.pairplot(data=df, hue="species", vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                     markers=["o", "^", "x"])
    g.fig.subplots_adjust(top=0.1)
    g.fig.suptitle('Iris DataSet', fontsize=16)
    plt.show()


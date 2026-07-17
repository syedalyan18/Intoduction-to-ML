import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(z):
    return 1/(1+np.exp(-z))

z=np.linspace(-10,10,100)
sigmoid_values=Sigmoid(z)

plt.plot(z,sigmoid_values)
plt.title("Sigmoid Function")
plt.xlabel("Z")
plt.ylabel(";(Z")
plt.grid()
plt.show()

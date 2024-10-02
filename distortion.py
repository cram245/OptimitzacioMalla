import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, -np.sqrt(3.) / 3], [0, 2 * np.sqrt(3.) / 3]],dtype=float)

def calculaDistorsioTriangle(Xe):

    DPhi = [Xe[1] - Xe[0], Xe[2] - Xe[0]] * A

    return np.linalg.norm(DPhi, 'fro') ** 2 / (2 * np.abs(np.linalg.det(DPhi)))


def calculaDistorsioMalla(X, T):
    
    
    suma = 0
    for i in range(len(T)):
        Xe = X[T[i, :], :]
        suma += (calculaDistorsioTriangle(Xe) ** 2)
    
    return np.sqrt(suma)
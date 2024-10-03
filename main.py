import numpy as np
import matplotlib.pyplot as plt

from mesh            import X,T,Nint,plotMesh,dofsToCoords,coordsToDofs
from differentiation import derivadaNumerica,hessianaNumerica
from distortion      import calculaDistorsioMalla

#------------------------------------------------------------------------------
#
# Completar el codi per determinar la posicio dels nodes interiors que
# minimitza la distorsio de la malla. 
# Abans de fer servir el metode de Newton, cal completar la funcio
# calculaDistorsioMalla
#  ...
#  ...
#  ...
#
#------------------------------------------------------------------------------

# Plot initial mesh configuration
plotMesh(X,'Initial mesh')

res = calculaDistorsioMalla(X, T)
print('Distorsio inicial: ',res)

def calculaResidu(y, T):
    # Convertir el vector y a coordenadas de los vértices interiores
    X = dofsToCoords(y)
    
    # Calcular el residuo como el gradiente de la distorsión
    residu = derivadaNumerica(lambda x: calculaDistorsioMalla(dofsToCoords(x), T), y)
    
    return residu

def calculaJacobiana(y, T):
    # Calcular la Jacobiana numéricamente
    jacobiana = hessianaNumerica(lambda x: calculaDistorsioMalla(dofsToCoords(x), T), y)
    
    return jacobiana

def newtonRaphson(y0, T, tol=5e-8, max_iter=100):
    y = y0
    errors = []
    for i in range(max_iter):
        residu = calculaResidu(y, T)
        

        if np.linalg.norm(residu) < tol:
            print(f'Convergencia alcanzada en {i} iteraciones')
            return y, errors
        jacobiana = calculaJacobiana(y, T)

        if i == 0:
            print('residu 0: ', residu[0])
            print('Jacobiana 0,0: ', jacobiana[0][0])

        delta_y = np.linalg.solve(jacobiana, -residu)
        errors.append(np.log10(np.linalg.norm(-delta_y)/np.linalg.norm(y + delta_y)))
        y = y + delta_y
        
    print('No se alcanzó la convergencia')
    return y, errors

y = coordsToDofs(X)  # Convertir las coordenadas iniciales a vector de incógnitas
y_opt, errors = newtonRaphson(y, T)
plt.plot(errors)
plt.title('Gràfica de Convergència')
plt.xlabel('Iteració')
plt.ylabel('Log(Error)')

plt.grid(True)
plt.show()


# Convert you vector with the dofs back to mesh coordinates
X = dofsToCoords(y_opt)
print(calculaDistorsioMalla(X, T))

plotMesh(X, 'Solució')
X = dofsToCoords(np.ones(y_opt.shape))


    


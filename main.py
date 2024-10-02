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

# Convert your mesh coordinates to a vector with the dofs
y = coordsToDofs(X)
raise Exception('CODE HERE THE MESH OPTIMIZATION')

# Convert you vector with the dofs back to mesh coordinates
X = dofsToCoords(y)
# X = dofsToCoords(np.ones(y.shape))


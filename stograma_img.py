import matplotlib.pyplot as plt
import numpy as np

# Lê a imagem
matriz = plt.imread("input/pb_cpf.png")   # coloque o nome do arquivo correto
#Mostra a imagem
plt.figure(1)
plt.imshow(matriz,cmap ='gray')
plt.show()

h = np.zeros(256)
for i in range (matriz.shape[0]):
    for j in range(matriz.shape[1]):
        h[matriz[i,j]]+=1


eixo_x=list(range(256))
plt.figure(2)
plt.bar(eixo_x, h)
plt.show()


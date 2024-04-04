import cv2
import numpy as np

imagen=cv2.imread("ruido.jpg",0)

m,n=imagen.shape

imagen_filtrada=np.zeros([m,n])

for i in range(1,m-1):
    for j in range(1,n-1):
        vecindario=[
            imagen[i-1,j-1],imagen[i-1,j],imagen[i-1,j+1],
            imagen[i,j-1],imagen[i,j],imagen[i,j+1],
            imagen[i+1,j-1],imagen[i+1,j],imagen[i+1,j+1]
        ]
        vecindario.sort()
        imagen_filtrada[i,j]=vecindario[4]

cv2.imshow('Original',imagen)
cv2.imshow('Mediana',imagen_filtrada.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
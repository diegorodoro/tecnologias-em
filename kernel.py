import cv2
import numpy as np

imagen=cv2.imread("ruido.jpg",0)

m,n=imagen.shape

imagen_convolucion=np.zeros([m,n])

#kernel de realce
kernel=np.array([[0,-1,0],
                 [-1,5,-1],
                 [0,-1,0]])

#bordes verticales
# kernel=np.array([[-1,0,1],
#                  [-2,0,2],
#                  [-1,0,1]])

#bordes horizontales
# kernel=np.array([[-1,-2,-1],
#                  [0,0,0],
#                  [-1,2,1]])

# for i in range(1,m-1):
#     for j in range(1,n-1):
#         sumatoria=  imagen[i-1,j-1]*kernel[0][0]+imagen[i-1,j]*kernel[0][1]+imagen[i-1,j+1]*kernel[0][2]+ \
#                     imagen[i,j-1]*kernel[1][0]+imagen[i,j]*kernel[1][1]+imagen[i,j+1]*kernel[1][2]+\
#                     imagen[i+1,j-1]*kernel[2][0]+imagen[i+1,j]*kernel[2][1]+imagen[i+1,j+1]*kernel[2][2]
#         imagen_convolucion[i,j]=sumatoria/9

imagen_convolucion2=cv2.filter2D(imagen,-1,kernel)

cv2.imshow('Original',imagen)
cv2.imshow('Mediana',imagen_convolucion2.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
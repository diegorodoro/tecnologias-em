import cv2
import numpy as np

#asi se le en colores, el indice 3 es el canal de colores
imagen=cv2.imread("puente.jpg")

nueva_img=np.zeros((190,215,3),dtype=np.uint8)
m,n=imagen.shape[:2]

for i in range(m):
    for j in range(n):
        if i >=420 and i<=610:
            if j>=385 and j<=600:
                nueva_img[i-420-1,j-385-1,:]=imagen[i,j,:]


cv2.rectangle(imagen,(420,385),(610,600),(0,255,0),3)

cv2.imshow('imagen',imagen.astype(np.uint8))
cv2.imshow('nueva img',nueva_img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


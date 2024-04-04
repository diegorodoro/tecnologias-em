import cv2
import numpy as np

imagen=cv2.imread("puente.jpg",0)

#bordes verticales
vertical=np.array([[-1,0,1],
                 [-2,0,2],
                 [-1,0,1]])

#bordes horizontales
horizontal=np.array([[-1,-2,-1],
                 [0,0,0],
                 [1,2,1]])

ver=cv2.filter2D(imagen,-1,vertical)
hor=cv2.filter2D(imagen,-1,horizontal)

imagen_convolucion=cv2.add(ver,hor)

cv2.imshow('Original',imagen)
cv2.imshow('vertical',ver.astype(np.uint8))
cv2.imshow('horizontal',hor.astype(np.uint8))
cv2.imshow('combinacion',imagen_convolucion.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()
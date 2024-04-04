import cv2
import numpy as np

#asi se le en colores, el indice 3 es el canal de colores
imagen=cv2.imread("puente.jpg")
m,n=imagen.shape[:2]

print(f"Dimensiones: {m}x{n}")
centro=(m//2,n//2)
print(f"Centro: {centro}")

matrix_rot_90=cv2.getRotationMatrix2D(centro,15,1.0)
img_rot_90=cv2.warpAffine(imagen,matrix_rot_90,(m,n))

img_flip_hor=cv2.flip(imagen,1)
img_flip_ver=cv2.flip(imagen,0)

cv2.imshow('Puente-original',imagen)
cv2.imshow('Puente-rot 90',img_rot_90)
cv2.imshow('Puente-flip hor',img_flip_hor)
cv2.imshow('Puente-flip ver',img_flip_ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
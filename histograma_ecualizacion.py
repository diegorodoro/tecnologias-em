import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen=cv2.imread("black_white.jpg",0)

plt.figure()
plt.title('Histograma de colores')
plt.xlabel('Intensidad')
plt.ylabel('Cantidad de pixeles')

histograma=cv2.calcHist([imagen], [0],None,[256],[0,256])
plt.plot(histograma)

plt.xlim([0,256])

# Ecualizacion puntual
# img_eq=cv2.equalizeHist(imagen)
# hist_eq=cv2.calcHist([img_eq], [0],None,[256],[0,256])
# plt.plot(hist_eq)

#ecualizacion adaptativa
umbral=2.0
tile=8

clahe=cv2.createCLAHE(clipLimit=umbral,tileGridSize=(tile,tile))
img_eq=clahe.apply(imagen)
hist_clahe=cv2.calcHist([img_eq], [0],None,[256],[0,256])
plt.plot(hist_clahe)

plt.show()

cv2.imshow('Original',imagen)
cv2.imshow('Ecualizada',img_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()
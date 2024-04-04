# import cv2
# import numpy as np

# imagen = cv2.imread("amongus.jpg")

# nueva_imagen = np.zeros((190, 215, 3), dtype=np.uint8)
# nueva_imagen[:, :, :] = (0,0,0)
# m,n = nueva_imagen.shape[:2]

# # for i in range(nueva_imagen.shape[0], nueva_imagen.shape[1]):
# #     nueva_imagen[i, i, :] = (0,0,255)  

# # for i in range(m):
# #     for j in range(n):
# #         nueva_imagen[j,j :] = (0,0,255)  

# for i in range(m):
#     for j in range(n):
#         if i >= 300 and i <= 500:
#             if j >= 300 and j <= 500:
#                 # imagen[i,j:] = 0
#                 nueva_imagen[i-300-1, j-300-1, :] = imagen[i,j,:]

# # imagen_reducida = cv2.resize(imagen, (n//5,m//5))
# # imagen_grande = cv2.resize(imagen, (n*5,m*5))

# cv2.imshow('AmongUs - Original', imagen)
# cv2.imshow('AmongUs - Original', nueva_imagen)
# # cv2.imshow('AmgounUs - Reducido', imagen_reducida)
# # cv2.imshow('AmgounUs - Grande', imagen_grande)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # SOLUCIONAR EL RECORTE 

import cv2
import numpy as np

imagen = cv2.imread("amongus.jpg")

x_start, y_start = 460, 550
x_end, y_end = 700, 650

imagen_recortada = imagen[y_start:y_end, x_start:x_end]


imagen_con_negro = imagen.copy()
imagen_con_negro[y_start:y_end, x_start:x_end] = 0


cv2.imshow('AmongUs - Original', imagen)
cv2.imshow('AmongUs - Recortada', imagen_recortada)
cv2.imshow('AmongUs - Modificacion', imagen_con_negro)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg

PATH_IMAGE = 'amongus.jpg' 

imagen = cv2.imread(PATH_IMAGE, 0)
m,n = imagen.shape[:2]

print(f"Dimensiones:{m}x{n}")
centro = (m // 2, n //2)
print(f"Centro: {centro}")

##historgrama 

histograma = cv2.calcHist([imagen], [0], None, [255], [0, 255])

plt.title('Histograma')
plt.xlabel('intensidad')
plt.ylabel('cantidad de pixeles(frecuencia)')
#plt.savefig['histograma.png']
plt.plot(histograma)
plt.show()


cv2.imshow('original', imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()
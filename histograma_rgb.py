import cv2
import numpy as np
import matplotlib.pyplot as plt

#Hola

PATH_IMAGE = 'amongus.jpg' 

imagen = cv2.imread(PATH_IMAGE)
# Convertir la imagen de BGR a RGB
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Separar los canales de la imagen
canal_r, canal_g, canal_b = cv2.split(imagen_rgb)

# Crear histogramas para cada canal
hist_r = cv2.calcHist([canal_r], [0], None, [255], [0, 255])
hist_g = cv2.calcHist([canal_g], [0], None, [255], [0, 255])
hist_b = cv2.calcHist([canal_b], [0], None, [255], [0, 255])

# Graficar los histogramas
plt.figure(figsize=(8, 6))
plt.title('Histograma de los Canales RGB')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.plot(hist_r, color='red', label='Rojo')
plt.plot(hist_g, color='green', label='Verde')
plt.plot(hist_b, color='blue', label='Azul')

plt.grid(True)
plt.show()

def sobreexponer(imagen, factor=100):
    # Aumentar los valores de píxeles para sobreexposición
    imagen_sobreexpuesta = cv2.add(imagen,np.ones(imagen.shape,dtype=np.uint8)*factor)
    return imagen_sobreexpuesta

def subexponer(imagen, factor=200):
    # Disminuir los valores de píxeles para subexposición
    imagen_subexpuesta = cv2.subtract(imagen,np.ones(imagen.shape,dtype=np.uint8)*factor)
    return imagen_subexpuesta

# Sobreexponer la imagen
imagen_sobreexpuesta = sobreexponer(imagen_rgb)

# Subexponer la imagen
imagen_subexpuesta = subexponer(imagen_rgb)

# Mostrar las imágenes original, sobreexpuesta y subexpuesta
plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
plt.imshow(imagen_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imagen_sobreexpuesta)
plt.title('Imagen Sobreexpuesta')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imagen_subexpuesta)
plt.title('Imagen Subexpuesta')
plt.axis('off')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

model=tf.keras.models.load_model('fashion_model.h5')


class_names=['T-shirt/top','Trouser',
             'Pullover','Dress',
             'Coat','Sandal',
             'Shirt','Sneaker',
             'Bag','Ankle boot']

#Cargar la imagen 
img=Image.open('test_1.jpg')
img=img.convert('L')
img=img.resize((28,28))
img=Image.fromarray(255-np.array(img))
img=np.array(img)
img=img/255.0
img=img.reshape(1,28,28)

prediction=model.predict(img)
label_pred=class_names[np.argmax(prediction)]

plt.imshow(img[0],cmap="binary_r")
plt.title(f'Predict: {label_pred}')
plt.show()
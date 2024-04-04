import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

n=random.randint(0,len(x_train))

# plt.imshow(x_train[n],cmap='binary_r')
# plt.title(f'n: {n}')
# plt.xlabel(f"Class: {class_names[y_train[n]]}")
# plt.show()


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i],cmap='binary_r')
#     plt.xlabel(class_names[y_train[i]])

# plt.show()

#normalizacion de los datos, van de 0 a 255, se pasan de 0 a 1
x_train,x_test=x_train/255,x_test/255

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64,activation='relu'))

model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_model=model.fit(x_train,y_train,epochs=20,validation_data=(x_test,y_test))

test_loss,test_acc=model.evaluate(x_test,y_test)

print(f'Loss:{test_loss}, Acurracy: {test_acc}')

plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

model.save("fashion_model.h5")
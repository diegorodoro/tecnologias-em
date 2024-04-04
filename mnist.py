import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist=tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
n=3

# plt.imshow(x_train[n], cmap='gray')
# plt.xlabel(y_train[n])
# plt.show()

#normalizar datos
x_train,x_test=x_train/255,x_test/255

#modelo
model = tf.keras.models.Sequential([
    # de matriz, a un vector
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # capa oculta
    tf.keras.layers.Dense(128,activation='relu'),
    # elimina neuronas que no se activan
    tf.keras.layers.Dropout(0.2),
    # capa de salida
    tf.keras.layers.Dense(10,activation='softmax')
])

# compilas
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo...")
# recuperas historial de perdidas por epoca
model_history=model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

# despues de epocas, se evalua presicion
print(model.evaluate(x_test,y_test))


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title("Progeso de perdida en el entrenamiento")
plt.xlabel("Pérdida")
plt.ylabel("Epocas")
plt.show()

predict=model.predict(x_test)
n=random.randint(0,1000)
print(f'Original: {y_test[n]} Prediccion: {np.argmax(predict[n])}')
plt.imshow(x_test[n],cmap='binary_r')
plt.xlabel(f"Yo digo que es {np.argmax(predict[n])}")
plt.show()
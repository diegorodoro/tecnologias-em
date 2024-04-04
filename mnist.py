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

x_train,x_test=x_train/255,x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo...")
model_history=model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

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
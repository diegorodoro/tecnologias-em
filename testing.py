import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

model=tf.keras.models.load_model('fashion_model.h5')
mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

class_names=['T-shirt/top','Trouser',
             'Pullover','Dress',
             'Coat','Sandal',
             'Shirt','Sneaker',
             'Bag','Ankle boot']

n=random.randint(0,len(x_test))

x_test=x_test/255

predictions=model.predict(x_test)
#matriz de confusion
config_matrix=tf.math.confusion_matrix(y_test,np.argmax(predictions,axis=1) )

sns.heatmap(config_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Reales')
plt.show()

# plt.imshow(x_test[n],cmap='binary_r')
# plt.xlabel(f'Real: {class_names[y_test[n]]}, Prediccion: {class_names[np.argmax(predictions[n])]}')
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(n,n+4):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i],cmap='binary_r')
#     plt.xlabel(f'Real: {class_names[y_test[n]]}, 
#                Prediccion: {class_names[np.argmax(predictions[n])]}')
# plt.show()
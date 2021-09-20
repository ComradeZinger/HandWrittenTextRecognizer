import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
import PIL.ImageOps  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(392, activation='relu'),
    Dense(196, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
# model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.1)
model.evaluate(x_test, y_test_cat)

# n = 29
# x = np.expand_dims(x_test[n], axis=0)

imgArr = ['1.png','3.png', '4.png', '5.png', '6.png']


def testImgArray (element) :
    img = Image.open(element)

    inverted_image = PIL.ImageOps.invert(img)

    inverted_image.save('new_name.png')
    image = cv2.imread('new_name.png', 0)

    x = np.expand_dims(image, axis=0)

    res = model.predict(x)

    # print(image)

    # print(res)
    print("Результат: ")
    print(np.argmax(res))
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


for element in imgArr:
    testImgArray(element)
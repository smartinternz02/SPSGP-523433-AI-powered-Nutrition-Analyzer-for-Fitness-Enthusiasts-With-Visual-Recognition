# -*- coding: utf-8 -*-
"""AI Assignment 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CjtCj2EgTFtJXtFxjqb9HH8lBdbUUjHa
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np

!unzip "/content/drive/MyDrive/archive.zip"

train_gen1 = ImageDataGenerator(rescale=(1./255),horizontal_flip=True,shear_range=0.2)
test_gen1 = ImageDataGenerator(rescale=(1./255))  #--> (0 to 255) convert to (0 to 1)


train1 = train_gen1.flow_from_directory('/content/train_data/train_data',
                                      target_size=(120, 120),color_mode = 'rgb',
                                      class_mode='categorical',
                                      batch_size=32)
test1 = test_gen1.flow_from_directory('/content/test_data/test_data',
                                    target_size=(120, 120),color_mode = 'rgb',
                                      class_mode='categorical',
                                      batch_size=32)

train1.class_indices

from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(Convolution2D(20,(3,3),activation='relu',input_shape=(120, 120, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(45,activation='relu'))
model.add(Dense(16,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train1,batch_size=32 ,validation_data=test1,epochs=10)

model.save('BirdsId.h5')

output = ['blasti',
 'bonegl',
 'brhkyt',
 'cbrtsh',
 'cmnmyn',
 'gretit',
 'hilpig',
 'himbul',
 'himgri',
 'hsparo',
 'indvul',
 'jglowl',
 'lbicrw',
 'mgprob',
 'rebimg',
 'wcrsrt']

# Testing 1
img1 = image.load_img('/content/Eagle.jpg',target_size=(120,120))
img1 = image.img_to_array(img1)
img1 = np.expand_dims(img1,axis=0)
pred = np.argmax(model.predict(img1))
print(pred)

print(output[pred])

# Testing 2
img2 = image.load_img('/content/crow.jpg',target_size=(120,120))
img2 = image.img_to_array(img2)
img2 = np.expand_dims(img2,axis=0)
pred = np.argmax(model.predict(img2))
print(pred)
print(output[pred])

# Testing 3
img3 = image.load_img('/content/owl.jpg',target_size=(120,120))
img3 = image.img_to_array(img3)
img3 = np.expand_dims(img3,axis=0)
pred = np.argmax(model.predict(img3))
print(pred)
print(output[pred])

model = Sequential()
model.add(Convolution2D(12,(3,3),activation='relu',input_shape=(120, 120, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(24,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(36,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(62,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(34,activation='relu'))
model.add(Dense(16,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train1,batch_size=32 ,validation_data=test1,epochs=10)

# Testing 1
img1 = image.load_img('/content/Eagle.jpg',target_size=(120,120))
img1 = image.img_to_array(img1)
img1 = np.expand_dims(img1,axis=0)
pred = np.argmax(model.predict(img1))
print(pred)

print(output[pred])

# Testing 2
img2 = image.load_img('/content/crow.jpg',target_size=(120,120))
img2 = image.img_to_array(img2)
img2 = np.expand_dims(img2,axis=0)
pred = np.argmax(model.predict(img2))
print(pred)
print(output[pred])

# Testing 3
img3 = image.load_img('/content/owl.jpg',target_size=(120,120))
img3 = image.img_to_array(img3)
img3 = np.expand_dims(img3,axis=0)
pred = np.argmax(model.predict(img3))
print(pred)
print(output[pred])

"""VGG16"""

train_gen = ImageDataGenerator(rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory('/content/train_data/train_data',
                                      target_size=(224,224),
                                      batch_size=32,
                                      class_mode='categorical')

test = test_gen.flow_from_directory('/content/test_data/test_data',
                                      target_size=(224,224),
                                      batch_size=32,
                                      class_mode='categorical')

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


vgg = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))

for layer in vgg.layers:
  print(layer)

for layer in vgg.layers:
  layer.trainable=False

x = Flatten()(vgg.output)

prediction = Dense(16,activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)


model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(train,validation_data=test,epochs=4,steps_per_epoch=len(train),validation_steps=len(test))

img4 = image.load_img('/content/myn.jpg',target_size=(224,224))
img4 = image.img_to_array(img4)
img4 = np.expand_dims(img4,axis=0)

pred = np.argmax(model.predict(img4))
print(pred)
print(output[pred])
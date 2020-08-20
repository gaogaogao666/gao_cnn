import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, Flatten
from tensorflow.keras import Model
import os

cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class CNN(Model):
  def  __init__(self):
    super(CNN,self).__init__()
    self.c1 = Conv2D(filters=6,kernel_size=(5,5),padding='same')
    self.b1 = BatchNormalization()
    self.a1 = Activation('relu')
    self.p1 = MaxPool2D(pool_size=(2,2),strides=2,padding='same')
    self.flatten = Flatten()
    self.d1 = Dense(128,activation='relu')
    self.d2 = Dense(10,activation='softmax')
  def call(self,x):
    x = self.c1(x)
    x = self.b1(x)
    x = self.a1(x)
    x = self.p1(x)
    x = self.flatten(x)
    x = self.d1(x)
    y = self.d2(x)
    
    return y
model = CNN()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()

                

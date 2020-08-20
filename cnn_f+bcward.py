import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, Flatten
from tensorflow.keras import Model
import os


def preprocess(x,y):
    x= tf.cast(x,tf.float32)/255.0
    x= tf.expand_dims(x,axis=-1)
    y= tf.one_hot(y,depth=10)
    return x,y

(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x.shape,y.shape,x.min(),x.max())

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(100)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(100)

#cifar10 = tf.keras.datasets.cifar10
#(x,y),(x_test,y_test)=cifar10.load_data()


#train_db = tf.data.Dataset.from_tensor_slices((x,y))
#train_db = train_db.shuffle(1000).map(preprocess).batch(100)

#test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
#test_db = test_db.shuffle(1000).map(preprocess).batch(100)


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

optimizer = tf.optimizers.Adam(lr=0.01) 
acc_metric= tf.metrics.Accuracy()
crossentropy= tf.losses.CategoricalCrossentropy(from_logits=True)

epochs=5

for epoch in range(epochs):

    for step,(x,y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            out = model(x)
            #print(out)
            loss = crossentropy(y,out)

        acc_metric.update_state(tf.argmax(y,axis=1),tf.argmax(out,axis=1))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step %100 ==0:
            print("epochs",epoch,"step:",step,"loss",float(loss),"train accuracy:",acc_metric.result().numpy())
            acc_metric.reset_states()

            print("exporting...")
            #model.save('model.h5')
            print("complete")



#model.compile(optimizer=optimizer,loss=loss,metrics=['sparse_categorical_accuracy'])
#model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()

                

import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,losses,metrics,datasets


# 预处理函数
def preprocess(x,y):
    
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.expand_dims(x, axis=-1)
    y = tf.one_hot(y, depth=10)
    return x,y


# 加载数据
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape,y.shape,x.min(),x.max())

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(100)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(100)


# 创建网络模型
network = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, padding='SAME', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Conv2D(16, kernel_size=2, strides=1, padding='SAME', activation='relu'),
    

    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation=None)
])
network.build(input_shape=[None,28,28,1])
network.summary()


# 创建优化器类、计量器类和损失函数类
optimizer = optimizers.Adam(lr=0.01)
acc_metric = metrics.Accuracy()
crossentropy = losses.CategoricalCrossentropy(from_logits=True)

epochs = 1

for epoch in range(epochs):
    
    for step, (x,y) in enumerate(train_db):
        
        with tf.GradientTape() as tape:
        
            out = network(x)
            loss = crossentropy(y, out)
    
        acc_metric.update_state(tf.argmax(out, axis=1), tf.argmax(y, axis=1))
        
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
        
        if step % 100 == 0:
            print("epochs:", epoch, "step:", step, "loss:", float(loss),
            "train accuracy:", acc_metric.result().numpy())
            acc_metric.reset_states()
            
            print("Exporting saved model..")
            network.save('model.h5')
            print("Complete export saved model.")

            
# 测试加载模型            
del network
print("Loading saved model..")
network = tf.keras.models.load_model('model.h5')
print("Complete load saved model.")
print("Test the model..")

for step, (x_test, y_test) in enumerate(test_db):
    
    out = network(x_test)
    
    acc_metric.update_state(tf.argmax(out, axis=1), tf.argmax(y_test, axis=1))
    
    if step % 10 == 0:
        print("step:", step, "test accuracy:", acc_metric.result().numpy())
        acc_metric.reset_states()

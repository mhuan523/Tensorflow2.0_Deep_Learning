# coding: utf-8

import tensorflow as tf

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os

tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert (tf.__version__.startswith('2.'))


def save_images(imgs, name):
    new_im = PIL.Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = PIL.Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


h_dim = 20
batchsz = 512
lr = 1e-3

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train.astype(np.float32) / 255., X_test.astype(np.float32) / 255.
X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(batchsz * 5).batch(batchsz)
X_test = tf.data.Dataset.from_tensor_slices(X_test).shuffle(batchsz * 5).batch(batchsz)


class AutoEncoder(keras.Model):
    """AutoEncoder"""

    def __init__(self):
        super(AutoEncoder, self).__init__()
        "encoder"
        self.encoder = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(h_dim),
        ])

        "decoder"
        self.decoder = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat


model = AutoEncoder()
model.build(input_shape=(None, 784))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)
os.makedirs("ae_images", exist_ok=True)

for epoch in range(100):

    for step, x in enumerate(X_train):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        # evaluation
        x = next(iter(X_test))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, '{}/rec_epoch_{}.png'.format("ae_images", epoch))

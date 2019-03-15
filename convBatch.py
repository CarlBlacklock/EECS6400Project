import tensorflow as tf

import numpy as np

from tensorflow.keras import layers

class convBatch(tf.keras.Model):
    def __init__(self, filters, size, strides, apply_batchnorm=True):
        super(convBatch, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.00)
        
        self.conv1 = layers.Conv2D(filters, (size, size), strides=(strides, strides), padding='same', kernel_initializer=initializer, use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization() 
    def call(self, x, training):
        y = self.conv1(x)
        if self.apply_batchnorm:
            y = self.batchnorm(y, training=training)
        return y
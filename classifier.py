import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from convBatch import convBatch
import math
def myAct(x):
    greater_cond = tf.greater(x, 1.0)
    res = tf.where(greater_cond, 1.0 + 0.001*x, x)
    less_cond = tf.less(x, -1.0)
    res = tf.where(less_cond, -1.0+0.001*x, res)
    return res

class classifier(tf.keras.Model):
    def __init__(self, num_classes, use_softmax=False, use_myAct = True):
        super(classifier, self).__init__()
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
        self.use_softmax = use_softmax
        self.num_classes = num_classes
        self.conv1 = convBatch(8, 5, 1)
        if use_myAct:
            self.act = layers.Lambda(myAct)
        else:
            self.act = layers.ReLU()
        self.conv2 = convBatch(8, 3,1)
        self.conv3 = convBatch(8, 3,1)
        self.conv4 = convBatch(16, 3, 1)
        self.conv5 = convBatch(16, 3, 1)
        self.conv6 = convBatch(64, 3,1)
        self.conv7 = convBatch(128, 1,1)
        self.local_pool = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(256, kernel_initializer=initializer)
        #self.dense2 = layers.Dense(128, kernel_initializer=initializer)
        #self.dense3 = layers.Dense(64, kernel_initializer=initializer)
        if use_softmax:
            self.out = layers.Dense(num_classes, activation = 'softmax', kernel_initializer=initializer)
        else:
            self.out = layers.Dense(num_classes, activation = 'sigmoid', kernel_initializer=initializer)
        #self.batchNorm1 = layers.BatchNormalization()
        #self.batchNorm2 = layers.BatchNormalization()
        #self.batchNorm3 = layers.BatchNormalization()
        #self.batchNorm4 = layers.BatchNormalization()
        #self.batchNorm5 = layers.BatchNormalization()
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.act(x)
        #x = self.batchNorm1(x, training=training)
        x = tf.concat([x, inputs], -1)
        
        y = self.conv2(x, training=training)
        y = self.act(y)
        #y = self.batchNorm2(y, training=training)
        x = tf.concat([y, x], -1)
        
        y = self.conv3(x, training=training)
        y = self.act(y)
        #y = self.batchNorm3(y, training)
        x = tf.concat([y, x], -1)
        
        x = self.local_pool(x)
        y = self.conv4(x, training=training)
        y = self.act(y)
        x = tf.concat([y, x], -1)
        #x = self.batchNorm4(x, training)
        
       # x = self.local_pool(x)
        y = self.conv5(x, training=training)
        y = self.act(y)
        x = tf.concat([y, x], -1)
        #x = self.batchNorm5(x, training=training)
        
        x = self.local_pool(x)
        x = self.conv6(x, training=training)
        x = self.act(x)
        
        x = self.conv7(x, training=training)
        x = self.act(x)
        
        x = self.global_pool(x)
        
        x = self.dense1(x)
        x = self.act(x)
        #x= self.dense2(x)
        #x = self.act(x)
        #x = self.dense3(x)
        #x = self.act(x)
        x = self.out(x)
        return x
        
    def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

class convClassifierBase(tf.keras.Model):
    def __init__(self, growth_rate, dense_blocks, layers_per_block, final_features):
        super(convClassifierBase, self).__init__()
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
        self.dense_blocks = dense_blocks
        self.final_features = final_features
        self.layers_per_block = layers_per_block
        self.growth_rate = growth_rate
        self.act = layers.ReLU()
        self.local_pool = layers.AveragePooling2D(pool_size=(2,2), padding='same')
        self.global_pool = layers.GlobalMaxPooling2D()
        self.convLayers = []
        self.prepoolConvLayers = []
        #Add all the convolutional layers that make up the dense blocks
        for i in range(0, dense_blocks):
            for j in range(0, layers_per_block):
                with tf.name_scope('dense_block_{0}_layer_{1}'.format(i, j)):
                    self.convLayers.append(convBatch(growth_rate, 3, 1))
        for i in range(0, dense_blocks):
            with tf.name_scope('transition_layer_{0}'.format(i)):
                self.prepoolConvLayers.append(convBatch(math.floor(0.5*(growth_rate*layers_per_block)), 1, 1))
        #Add final layer that generates the output features
        with tf.name_scope('global_feature_layer'):
            self.finalConv = convBatch(final_features, 1, 1)
        
    def call(self, inputs, training=False):
        #Apply all blocks
        for i in range(0, self.dense_blocks):
            #Apply all layers in block
            for j in range(0, self.layers_per_block):
                x = self.convLayers[j + self.layers_per_block*i](inputs, training=training)
                x = self.act(x)
                inputs = tf.concat([x, inputs], -1)
            #All layers in block applied
            #Apply prepoolConvLayer
            inputs = self.prepoolConvLayers[i](inputs, training=training)
            inputs = self.act(inputs)
            #Pool and apply next block
            inputs = self.local_pool(inputs)
        #All dense blocks applied
        #Apply final conv layer
        x = self.finalConv(inputs, training=training)
        x = self.act(x)
        #Apply global pooling
        x = self.global_pool(x)
        return x
        
class multiCategoryClassifier(tf.keras.Model):
    def __init__(self, growth_rate, dense_blocks, layers_per_block, final_features):
        super(multiCategoryClassifier, self).__init__()
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
        self.convClassifierBase = convClassifierBase(growth_rate, dense_blocks, layers_per_block, final_features)
       
        self.genderClassifier = layers.Dense(2, kernel_initializer=initializer)
        self.ageClassifier = layers.Dense(7, kernel_initializer=initializer)
        #self.category_activations = []
        #for i in range(0, self.num_categories):
            #if category_list[i]['activation'] == 'sigmoid':
                #self.category_linear_layers.append(layers.Dense(category_list[i]['number_of_labels'], activation='sigmoid', kernel_initializer=initializer))
            #elif category_list[i]['activation'] == 'softmax':
                #self.category_linear_layers.append(layers.Dense(category_list[i]['number_of_labels'], activation='softmax', kernel_initializer=initializer))
            #else:
                #No activation
                #self.category_linear_layers.append(layers.Dense(category_list[i]['number_of_labels'], kernel_initializer=initializer))
    
    def call(self, inputs, training=False):
        x = self.convClassifierBase(inputs, training=training)
        #outputs = self.category_linear_layers[0](x)
        #for i in range(1, self.num_categories):
            #outputs = tf.concat([outputs, self.category_linear_layers[i](x)], -1)
        #outputs = self.category_linear_layers[0](x)
        return self.genderClassifier(x), self.ageClassifier(x)
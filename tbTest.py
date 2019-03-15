import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from classifier import multiCategoryClassifier
#from load_image_batch import load_batch


trainFile = ['train_photo_labels.txt']
testFile = ['test_photo_labels.txt']
image_dir = './Flickr_Photos_Full'
checkpoint_dir = './checkpoints/run5'
cache_dir = './cache'
batch_size = 16
test_batch_size = 16
number_of_categories = 2
category_list = [{'number_of_labels':2, 'activation':'none'},{'number_of_labels': 7, 'activation': 'none'}]
growth_rate = 8
dense_blocks = 3
layers_per_block = 6
final_features = 128

def init_graph():
    with tf.name_scope("classifier"):
        classifier = multiCategoryClassifier(growth_rate, dense_blocks, layers_per_block, final_features, number_of_categories, category_list)
    
    return classifier
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
  #image = tf.image.resize_images(image, [192, 192])
  #image = tf.cast(image, tf.float32)
  #image = image / 255.0  # normalize to [0,1] range
  #image = (2.0)*image - 1.0
    return image


  
def cast_image(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = (2.0)*image - 1.0
    return image
    
def cast_image_float(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image
    
def cast_image_range(image):
    image = (2.0)*image - 1.0
    return image
def image_augment(image, new_height, new_width):

    image = tf.image.resize_image_with_pad(image, new_height,new_width) 
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.125)
    image = tf.image.random_saturation(image, 0.0, 0.125)
    return image

def preprocess_and_augment_image_with_cast(image, new_height, new_width):
    image = preprocess_image(image)
    image = cast_image_float(image)
    image = image_augment(image, new_height, new_width)
    image = cast_image_range(image)
    #image = cast_image(image)
    return image
    
def preprocess_image_with_cast(image, new_height, new_width):
    image = preprocess_image(image)
    image = test_augment(image, new_height, new_width)
    image = cast_image(image)
    return image
    
def test_augment(image, new_height, new_width):
    image = tf.image.resize_image_with_pad(image, new_height, new_width)
    return image
  
def load_and_preprocess_image(path, label):
    image = tf.read_file(path)
    label = tf.cast(label, tf.int64)
    return preprocess_image(image), label

def load_train_batch(batch_path, batch_gen_label, batch_age_label):
    new_height = tf.random.uniform(shape=(), minval=64, maxval=512, dtype=tf.int32)
    #new_width = tf.random.uniform(shape=(), minval=64, maxval=512, dtype=tf.int32)
    new_width = tf.random.uniform(shape=(), minval=tf.maximum(64, new_height-tf.cast((tf.cast(new_height, tf.float32)*0.25), tf.int32)), maxval=tf.minimum(512, new_height+tf.cast((tf.cast(new_height, tf.float32)*1.25), tf.int32)), dtype=tf.int32)
    batch_list = tf.unstack(batch_path, num=batch_size, axis=0)
    image_batch = tf.expand_dims(preprocess_image_with_cast(tf.read_file(batch_list[0]), new_height, new_width), 0)
    for path in batch_list[1:]:
        image = tf.expand_dims(preprocess_and_augment_image_with_cast(tf.read_file(path), new_height, new_width), 0)
        image_batch = tf.concat([image_batch, image], 0)
    #image = tf.read_file(path)
    gen_label = tf.cast(batch_gen_label, tf.int64)
    age_label = tf.cast(batch_age_label, tf.int64)
    #image = preprocess_and_augment_image_with_cast(image)
    return image_batch, gen_label, age_label

def load_test(path, gen_label, age_label):
    #new_height = tf.random.uniform(shape=(), minval=64, maxval=512, dtype=tf.int32)
    #new_width = tf.random.uniform(shape=(), minval=tf.maximum(64, new_height-tf.cast((tf.cast(new_height, tf.float32)*0.25), tf.int32)), maxval=tf.minimum(512, new_height+tf.cast((tf.cast(new_height, tf.float32)*1.25), tf.int32)), dtype=tf.int32)
    #new_width = tf.math.floor(1.5*tf.cast(new_width, tf.float32))
    new_height = 128
    new_width = 128
    image = preprocess_image_with_cast(tf.read_file(path), new_height, new_width)
    gen_label = tf.cast(gen_label, tf.int64)
    age_label = tf.cast(age_label, tf.int64)
    #image = preprocess_and_augment_image_with_cast(image)
    return image, gen_label, age_label
  
def split_line(line):
    #splits a line into the filename and features
    line = str(line, encoding='utf-8')
    #print(line)
    line = line.rstrip('\n')
    line = line.split('\t')
    #print(line)
    source = '{0}/{1}'.format(image_dir,line[0])
    label = [int(line[1]),int(line[2])]
    return source, label

def split_label(source, label):
    return source, label[0], label[1]

#Define TensorFlow Dataset
#AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices(trainFile)
train_dataset = train_dataset.flat_map(lambda filename: (tf.data.TextLineDataset(filename).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

train_dataset = train_dataset.map(lambda line: tuple(tf.py_func(split_line, [line], [tf.string, tf.int32])))
train_dataset = train_dataset.map(split_label)
#train_dataset = train_dataset.cache(filename='./cache/train-data')
train_dataset = train_dataset.shuffle(buffer_size=80)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size) 

train_dataset = train_dataset.map(load_train_batch)
train_dataset = train_dataset.prefetch(buffer_size=2*batch_size)
train_iterator = train_dataset.make_initializable_iterator()


test_dataset = tf.data.Dataset.from_tensor_slices(testFile)
test_dataset = test_dataset.flat_map(lambda filename: (tf.data.TextLineDataset(filename).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
#test_dataset = test_dataset.take(6000)
test_dataset = test_dataset.map(lambda line: tuple(tf.py_func(split_line, [line], [tf.string, tf.int32])))
test_dataset = test_dataset.map(split_label)
test_dataset = test_dataset.map(load_test)
test_dataset = test_dataset.cache(filename='./cache/test-data') 
#For testing we want to iterate over the entire test set once per epoch to calculate accuracy
#So we batch then repeat which avoids any data element being counted more than once
test_dataset = test_dataset.batch(test_batch_size)
test_dataset = test_dataset.repeat()


test_dataset = test_dataset.prefetch(buffer_size=2*test_batch_size)
test_iterator = test_dataset.make_initializable_iterator()

#train_image_batch, train_label_batch = train_iterator.get_next()

def getLoss(genLabel, ageLabel, genderPredictions, agePredictions):
    #genLoss = tf.losses.sparse_softmax_cross_entropy(tf.slice(labels, [0, 0], [-1, 1]), tf.slice(predictions, [0, 0], [-1, 2]))
    genLoss = tf.losses.sparse_softmax_cross_entropy(genLabel, genderPredictions)
    #ageLoss = tf.losses.sparse_softmax_cross_entropy(tf.slice(labels, [0, 1], [-1, -1]), tf.slice(predictions, [0, 2], [-1, -1]))
    ageLoss = tf.losses.sparse_softmax_cross_entropy(ageLabel, agePredictions)
    loss_val = genLoss + ageLoss
    return loss_val

    
def getAcc(genLabel, ageLabel, genderPredictions, agePredictions):
    genAcc = 0.0
    ageAcc = 0.0
    #offset = 0
    #genCorrect = tf.equal(tf.argmax(tf.slice(predictions, [0, 0], [-1, 2]), 1), tf.slice(labels, [0, 0], [-1, 1]))
    
    #ageCorrect = tf.equal(tf.argmax(tf.slice(predictions, [0, 2], [-1, -1]), 1), tf.slice(labels, [0, 1], [-1, -1]))
    genAcc = tf.reduce_mean(tf.cast(genCorrect, tf.float32))
    #ageAcc = tf.reduce_mean(tf.cast(ageCorrect, tf.float32))
    return genAcc
    #, ageAcc
optimizer = tf.train.AdamOptimizer()
with tf.name_scope("classifier"):
        classifier = multiCategoryClassifier(growth_rate, dense_blocks, layers_per_block, final_features)

        
#input_image, target_image = load_batch(image_dir, batch_size, 1)
#input = tf.placeholder(tf.float32, [None, None, None, 3], "gen_input")
#target = tf.placeholder(tf.float32, [None, None, None, 3], "target_image")
train_image_batch, train_gen_label, train_age_label = train_iterator.get_next()
genderPredictions, agePredictions = classifier(train_image_batch, training=True)

loss_val = getLoss(train_gen_label, train_age_label, genderPredictions, agePredictions)
#sanity_check = tf.print(train_image_batch.shape, train_gen_label, train_age_label)
test_image_batch, test_gen_label, test_age_label = test_iterator.get_next()
test_predictions_gender, test_predictions_age = classifier(test_image_batch, training=False)
#gen_loss = generator_loss(disc_gen_output, y, target_image)[0]
#gen_opt = tf.train.AdamOptimizer(name="gen_opt")
#disc_opt = tf.train.AdamOptimizer(name="disc_opt")
train = optimizer.minimize(loss_val, var_list=classifier.variables)
#train_disc = disc_opt.minimize(disc_loss, var_list=disc.variables)
#g_loss = tf.summary.scalar("generator loss", gen_loss)
c_loss = tf.summary.scalar("classification loss", loss_val)
#ageAcc_metric, ageAcc_update = tf.metrics.accuracy(tf.slice(test_label_batch, [0, 1], [-1, -1]), tf.argmax(tf.slice(test_predictions, [0, 2], [-1, -1]), 1), name='age_metric')
#genAcc_metric, genAcc_update = tf.metrics.accuracy(tf.slice(test_label_batch, [0, 0], [-1, 1]), tf.argmax(tf.slice(test_predictions, [0, 0], [-1, 2]), 1), name='gender_metric')
ageAcc_metric, ageAcc_update = tf.metrics.accuracy(test_age_label, tf.argmax(test_predictions_age,1), name='age_metric')
genAcc_metric, genAcc_update = tf.metrics.accuracy(test_gen_label, tf.argmax(test_predictions_gender,1), name='gender_metric')
acc_gen = tf.summary.scalar('gender accuracy', genAcc_metric)
acc_age = tf.summary.scalar('age accuracy', ageAcc_metric)
#merged = tf.summary.merge_all()
summary_image = tf.summary.image("example_train_image", train_image_batch, max_outputs=1)
merged_train_summary = tf.summary.merge([c_loss, summary_image])
merged_test_summary = tf.summary.merge([acc_gen, acc_age])

ageMetric_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='age_metric')
genMetric_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='gender_metric')

ageMetricInitializer = tf.variables_initializer(var_list=ageMetric_running_vars)
genMetricInitializer = tf.variables_initializer(var_list=genMetric_running_vars)

def run_sess():
    
    with tf.Session() as sess:
        options = tf.RunOptions()
        options.output_partition_graphs = True
        options.trace_level = tf.RunOptions.FULL_TRACE
        metadata = tf.RunMetadata()
        writer = tf.summary.FileWriter("./class_train/run5", sess.graph)
        #test_writer = tf.summary.FileWriter("./class_test")
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(classifier=classifier)
        #_ = sess.run(sanity_check)
        #sys.exit()
        for epoch in range(1, 1501):
            for batch in range(5):
                
            #if i % 10 == 0:
                #summary, acc, _ = sess.run([acc_sum, accuracy, sanity_check])
                #test_writer.add_summary(summary, i)
                #print('Accuracy at step %s: %s' % (i, acc))
                #if epoch == 1 and batch == 4:
                    #summary, _ = sess.run([merged_train_summary, train], options=options, run_metadata=metadata)
                    #writer.add_run_metadata(metadata, 'epoch%03d' % epoch)
                    #writer.add_summary(summary, epoch*5 + batch)
          
                
                if epoch % 10 == 0 and batch == 0:
                    summary, _ = sess.run([merged_train_summary, train])
                    writer.add_summary(summary, epoch*5 + batch)
                    checkpoint.save(file_prefix=checkpoint_prefix)
                else:
                    summary, _,= sess.run([c_loss,train])
                #writer.add_run_metadata(metadata, 'disc step %d' % i)
                    writer.add_summary(summary, epoch*5 + batch)
                #merged, _ = sess.run([g_loss, train_gen])
                #writer.add_run_metadata(metadata, 'gen step %d' % i)
                #writer.add_summary(merged, i)
            #At the end of each epoch calculate accuracy over test set
            #Reset metric running averages
            sess.run([ageMetricInitializer, genMetricInitializer])
            #sess.run(genMetricInitializer)
            for batch in range(10):
                sess.run([ageAcc_update, genAcc_update])
                #sess.run(genAcc_update)
            summary = sess.run(merged_test_summary)
            writer.add_summary(summary, epoch)
            print('completed epoch {0}'.format(epoch))
        checkpoint.save(file_prefix=checkpoint_prefix)    
        writer.close()
        
if __name__ == '__main__':
    run_sess()